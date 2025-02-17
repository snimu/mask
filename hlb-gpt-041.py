# Colab users, uncomment the following block to help clear out notebook state when re-running the cell.
"""
# don't forget these too:
# !pip3 install tiktoken
# If you don't have torch 2.0 on whatever environment you're using:
# !pip3 install --upgrade torch
try:
  _ = get_ipython().__class__.__name__
  ## we set -f below to avoid prompting the user before clearing the notebook state
  %reset -f
except NameError:
  pass ## we're still good
"""
from typing import Literal
import itertools
import argparse
import functools
from functools import partial
import subprocess

import einops
import zipfile
import math
import os

import torch
import torch.nn.functional as F
from torch import nn
import polars as pl

# This seems like one of the best choices right now for a fast/lightweight/simple tokenizer.
import tiktoken


################
# Introduction #
################

# This code was built from the ground up to support extremely rapid experimentation for solo researchers and small teams. It's meant to
# be hackable nearly anywhere with minimal effort/side effects, which is why you might see more of a flat layout. It's also quite fast.
#
# The codebase is specifically designed for single A100s for now, but may expand with more GPU support in the future, depending. I originally
# used Karpathy's nanoGPT as well as some of my other work as a reference when writing this, though this codebase is very much
# its own thing at this point.
#
# If you found this codebase useful or informative, please consider supporting me directly at https://www.patreon.com/tysam . If you'd like
# to speak about a contract or a consulting opportunity, feel free to reach out at hi [dot] re [dot] tysam [atsymbol] gmail [dot] com.
# I'd love to hear from you!
#
# Now, on with the code!


##############################
#      Hyperparameters       #
##############################

# Note: The automatic rescaling of hyperparameters based on batchsize/etc is currently a work in progress.
# This code assumes 40 GB-limit A100s for the scale-based hyperparameters, you may have to do some tinkering if you have a different setup.
# So far, most of the tested configs have been between ~46 M and 1.5B or so, and have done moderately well.

# This parameter determines the final size of the model. Roughly, num_model_params ~= model_scale * 49 M (# of params in the base model), but it scales nonlinearly. (#TODO is to make this more straight in the future)
# Model scales other than 1.0 are in alpha currently -- they should run okay, but are almost certainly not tuned efficiently yet! This should hopefully be addressed in a future update.
model_scale         = 1.0    # OOM-tested from ~.5ish (28 M) to 148 (~3 B). Sets the model size. One of the most important hyperparameters. Supports noninteger values (2.3, etc)
max_sequence_length = 1024   # Can go up or down. Mostly tested up to 1024, some models can avoid OOMs even with length 8192 (not really tested)
gpu_token_capacity  = 114688 # This is an amount that doesn't OOM on A100 at model_scale 1, length 1024. May need to change if you have a different GPU. Note: Hyperparameter tunings are currently based on the 40 GB limit of the A100.

# Approximates the amount of tokens the GPU can hold based upon the scale of the model (scaled somewhat conservatively to avoid most OOMs. May OOM in some weird edgecases.)
# Batchsize is determined automatically based upon the current sequence length and the rough token-capacity of the GPU for a given model.
tokens_per_batch_capacity  = math.floor(gpu_token_capacity / (1.52174 + .482 * model_scale**(.87)))

# We support fractional model factors, this picks dimensions that the A100 can efficiently use.
to_nearest_64 = lambda x: round(x/64) * 64


# The default model here below is roughly ~46M parameters or so.
hyp = {
    'opt': {
        'lr_mult': {
            'base': 2.62, # The base_lr itself is derived from a scaling equation fit to GPT-3 parameters. This multiplier impacts all parameters, including those in the default group
            'position_bias': 100.,
            'non_dot_products': 32.,
            'output_layer': 2.,
        },
        'weight_decay': 2.**4,     # This is the weight decay when the loss = 0., we approach it exponentially. Somewhat slows overfitting.
        'total_train_steps': 1000, # We can run effectively infinitely, but is 1000 by default for the inference demo. For infinite runs, you can use the saved checkpoints from disk.
        'microbatch': {            # The microbatch scheduler assumes a power law decay schedule for the grad norm, and adjusts the microbatch size (minimum 1) to enforce it.
            'sample_every': 5,     # Sampling grad norm can be a bit expensive, so we do it every n steps instead.
            'scale_lr': 1e-1,      # Microbatch update rate
        },
        'eval_every': 50,          # how many train iterations per eval round (we don't include eval time in our performance stats). Good to set to 10-20 for larger (~800M+ networks)
        'save_every_n_evals': 2,   # Good to set this low for larger networks
        'num_eval_tokens': 153600, # Total # tokens total to eval over, divided into max_sequence_length-long sequences
        'warmup_steps': 100,       # For training stability in the main body of the network. (#TODO: Investigate the warmup imact a bit more)
    },
    'net': {
        'residual_depth': to_nearest_64(384 * math.log2(1.+model_scale)),
        'qk_dim_div': 8,
        'expand_factor': 2,
        'num_blocks': round(8 * math.log2(1.+model_scale)),
    },
    'misc': {
        'num_tokens': 50304, # Rounded to the nearest value of 64 for efficiency
        'sequence_length': {
            'max': max_sequence_length,
            'initial': 32,      # Very short initial sequence length seems to help a lot
            'growth_steps': 80, # We double the sequence length during training every n steps up to the maximum
        },
        'device': 'cuda',
        'dtype': torch.bfloat16,
        'data_location': 'data.pt',
    }
}


def change_gpu_token_capacity(factor: float):
    global gpu_token_capacity
    gpu_token_capacity = factor * 114688


def change_model_scale(scale: float, depth: int | None = None, width: int | None = None):
    global model_scale, tokens_per_batch_capacity, hyp
    if depth is not None or width is not None:
        assert width is not None and depth is not None
        hyp['net']['residual_depth'] = to_nearest_64(width)
        hyp['net']['num_blocks'] = depth

        # Adapt model scale
        net = make_net(1)
        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        del net
        default_params = 46_009_736
        model_scale = num_params / default_params
        tokens_per_batch_capacity  = math.floor(gpu_token_capacity / (1.52174 + .482 * model_scale**(.87)))
        return
    
    model_scale = scale
    tokens_per_batch_capacity  = math.floor(gpu_token_capacity / (1.52174 + .482 * model_scale**(.87)))
    hyp['net']['residual_depth'] = to_nearest_64(384 * math.log2(1.+model_scale))
    hyp['net']['num_blocks'] = round(8 * math.log2(1.+model_scale))


#############################################
#                Dataloader                 #
#############################################

if not os.path.exists(hyp['misc']['data_location']):
    print("downloading data and tokenizing (1-2 min)")

    raw_data_source = 'https://wikitext.smerity.com/wikitext-103-raw-v1.zip'
    raw_data_cache = './data_raw/' # where to cache the data after downloading

    if not os.path.isfile(raw_data_cache):
        os.makedirs(raw_data_cache, exist_ok=True)

        # Needed due to the website 403-blocking python agents for download, it seems? Many thanks to Smerity for re-hosting these after the main files went down. <3 :')
        subprocess.run(["wget", raw_data_source, "-O", raw_data_cache+"data.zip"], stdout=subprocess.PIPE)

    with zipfile.ZipFile('data_raw/data.zip', 'r') as zip_ref:
        zip_ref.extractall('data_raw/')

    with open('data_raw/wikitext-103-raw/wiki.train.raw') as data_file:
        raw_train_data = data_file.read()

    with open('data_raw/wikitext-103-raw/wiki.valid.raw') as data_file:
        raw_eval_data = data_file.read()


    tokenizer = tiktoken.get_encoding("gpt2")
    raw_tokenized_train = tokenizer.encode_ordinary(raw_train_data)
    raw_tokenized_eval  = tokenizer.encode_ordinary(raw_eval_data)

    train_tokenized = torch.tensor(raw_tokenized_train, device=hyp['misc']['device'], dtype=torch.int) # int64 is likely overkill for the amount of tokens we have...
    eval_tokenized  = torch.tensor(raw_tokenized_eval,  device=hyp['misc']['device'], dtype=torch.int)

    data = {
        'train': train_tokenized,
        'eval': eval_tokenized
        }

    torch.save(data, hyp['misc']['data_location'])
    print("completed the tokenization process!")

else:
    ## This is effectively instantaneous, and takes us practically straight to where the dataloader-loaded dataset would be. :)
    ## So as long as you run the above loading process once, and keep the file on the disc it's specified by default in the above
    ## hyp dictionary, then we should be good. :)
    data = torch.load(hyp['misc']['data_location'])


########################################
#              Constants               #
########################################

with torch.no_grad():
    # Create the base arrays for the learnable linear positional bias. This helps save some memory consumption & processing time
    bias_range                    = torch.arange(-hyp['misc']['sequence_length']['max']+1, 1).to(hyp['misc']['device'], torch.bfloat16)
    position_bias_base_forward    = bias_range.unsqueeze(0) - bias_range.unsqueeze(1)
    position_bias_base_backward   = position_bias_base_forward.flip(0)
    position_bias_base            = position_bias_base_forward
    negative_infinity_matrix_base = torch.empty_like(position_bias_base).fill_(-float("inf"))
    causal_mask_forward = torch.tril(torch.ones((hyp['misc']['sequence_length']['max'], hyp['misc']['sequence_length']['max']), device=hyp['misc']['device'], dtype=torch.bool))
    causal_mask_backward = torch.tril(torch.ones((hyp['misc']['sequence_length']['max'], hyp['misc']['sequence_length']['max']), device=hyp['misc']['device'], dtype=torch.bool)).flip(0)
    causal_mask = causal_mask_forward
    causality: Literal["forward", "backward"] = "forward"


# Used in the dataloader to select indexes in a sequence. Preallocated for slight efficiency.
batch_index_offsets = torch.arange(0, hyp['misc']['sequence_length']['max']+1, dtype=torch.long, device=hyp['misc']['device'])


#############################################
#            Network Components             #
#############################################

class LatentAttentionBlock(nn.Module):
    """ Efficient fused latent-space attention block. Linear keys and queries, nonlinear values."""
    def __init__(self, num_dim, linear_value: bool, num_heads: int):
        super().__init__()
        # Layer dim parameters. Play around with these, there's likely some undiscovered stuff still!
        self.dim        = num_dim
        self.qk_dim     = self.dim//hyp['net']['qk_dim_div']
        self.v_dim      = num_dim
        self.expand_dim = num_dim * hyp['net']['expand_factor']
        self.linear_value = linear_value 
        self.num_heads = num_heads

        # Main layer weights
        self.norm    = nn.LayerNorm(self.dim, bias=False)
        self.expand  = nn.Parameter(.5 * 1./hyp['net']['residual_depth']**.5 * 1./hyp['net']['expand_factor']                               * torch.randn(2*self.qk_dim+2*self.expand_dim, self.dim))
        self.project = nn.Parameter(1. * 1./hyp['net']['residual_depth']**.5 * 1./hyp['net']['expand_factor'] * 1./hyp['net']['num_blocks'] * torch.randn((self.dim, self.expand_dim)))

        # Learnable linear positional encodings. Similar to but different than https://arxiv.org/abs/2108.12409
        # Has a high lr mult applied to it so that each layer can learn its own attention scale.
        self.position_bias_mult = nn.Parameter(torch.tensor(1., device='cuda'))

    def forward(self, x):
        residual = x

        # Make additive attention mask, scaled by a learned mult for the position bias (lets us learn dynamic attention ranges per layer as needed)
        attn_mask = torch.where(causal_mask[:x.shape[1], :x.shape[1]], F.softplus(self.position_bias_mult) * position_bias_base[:x.shape[1], :x.shape[1]], negative_infinity_matrix_base[:x.shape[1], :x.shape[1]])

        # Shared LayerNorm for linear layers and attention
        x = self.norm(x)

        # Fused into one kernel for memory+speed/etc
        query, key, linear, pre_gelu = F.linear(x, self.expand).split((self.qk_dim, self.qk_dim, self.expand_dim, self.expand_dim), dim=-1)

        # Compute GeGLU (one portion of the channels this will stay locally, another will become the nonlinear value for attention)
        geglu = linear * F.gelu(pre_gelu)

        # Partition between the input values and the v dim values
        if self.linear_value:
            geglu_local, _ = geglu.split((self.expand_dim-self.v_dim, self.v_dim), -1)
            _, geglu_attention_value = pre_gelu.split((self.expand_dim-self.v_dim, self.v_dim), -1)
        else:
            geglu_local, geglu_attention_value = geglu.split((self.expand_dim-self.v_dim, self.v_dim), -1)

        if self.num_heads > 1:
            query, key, geglu_local, geglu_attention_value = map(lambda x: einops.rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), (query, key, geglu_local, geglu_attention_value))


        # Compute attention. Something to note is that there are no attention heads here. This seemed to work a bit better, maybe due to not needing memory `.contiguous()` calls or similar
        attention = F.scaled_dot_product_attention(query, key, geglu_attention_value, attn_mask=attn_mask)

        if self.num_heads > 1:
            attention = einops.rearrange(attention, 'b h n d -> b n (h d)')
            geglu_local = einops.rearrange(geglu_local, 'b h n d -> b n (h d)')

        # Output linear layer
        out = F.linear(torch.cat([geglu_local, attention], dim=-1), self.project)

        # Add to residual
        x = residual + out

        return x


#############################################
#            Network Definition             #
#############################################

# This may seem like an odd way to define a network, but it's a bit easier to hack into/make quick changes than other methods
class SpeedyLangNet(nn.Module):
    def __init__(self, network_dict):
        super().__init__()
        self.net_dict = network_dict

    def forward(self, x):
        # Look up the input embeddings from the input tokens
        x = self.net_dict['embedding'](x)
        for block in range(hyp['net']['num_blocks']):
            x = self.net_dict['attn_layers'][block](x) # note: residuals are included in the block definitions for these layers
        x = self.net_dict['norm'](x)
        x = self.net_dict['outputs'](x)
        return x


def make_net(num_heads: int = 1):
    network_dict = nn.ModuleDict({
        # Add two special tokens for the forward and backward pass, even if they are unused
        'embedding': nn.Embedding(hyp['misc']['num_tokens']+2, hyp['net']['residual_depth'], scale_grad_by_freq=True),
        'attn_layers': nn.ModuleList([
            LatentAttentionBlock(
                hyp['net']['residual_depth'],
                linear_value=False,
                num_heads=num_heads,
            ) 
            for _ in range(hyp['net']['num_blocks'])]
        ),
        'norm': nn.LayerNorm(hyp['net']['residual_depth'], bias=False),
        'outputs': nn.Linear(hyp['net']['residual_depth'], hyp['misc']['num_tokens']+2, bias=False),
})
    net = SpeedyLangNet(network_dict)
    net = net.to(hyp['misc']['device'], torch.bfloat16)
    net.train()

    # Initialize the embedding and output matrixes, with weights scaled based upon the dimensionality of the network.
    torch.nn.init.normal_(net.net_dict['embedding'].weight.data, std=.25*1./hyp['net']['residual_depth']**.5)
    torch.nn.init.normal_(net.net_dict['outputs']  .weight.data, std=.5 *1./hyp['net']['residual_depth']**.5)

    return net


def remove_layers(net: SpeedyLangNet, n: int) -> SpeedyLangNet:
    if n >= len(net.net_dict["attn_layers"]):
        raise ValueError(f"n may not be greater than the number of blocks! {n=}, n_blocks={len(net.net_dict['attn_layers'])}")
    
    for _ in range(n):
        net.net_dict["attn_layers"].pop(-1)
    
    global hyp
    hyp['net']['num_blocks'] -= n

    return net


########################################
#          Training Helpers            #
########################################

# Get a single batch item. Currently used in the training loop
@torch.no_grad()
def get_batch(data_dict, key, batchsize, length):
    start_indexes     = torch.randint(len(data_dict[key])-length-1, (batchsize,), device=hyp['misc']['device']) # warning, completely random sampling, not a random derangement, that might help performance a bit!
    sequence_indexes  = start_indexes.unsqueeze(-1) + batch_index_offsets[:length+1].unsqueeze(0) # slice, as batch_index_offsets are pre-allocated to max length for efficiency
    sampled_sequences = torch.take_along_dim(data_dict[key], sequence_indexes.flatten(), dim=0).view(batchsize, length+1).long() # have to flatten and reshape due to take_along_dim being 1d
    return sampled_sequences


FW_TOKEN = torch.tensor(hyp['misc']['num_tokens'], device=hyp['misc']['device'], dtype=torch.int)
BW_TOKEN = FW_TOKEN + 1


@torch.no_grad()
def split_batch(sampled_sequences: torch.Tensor, direction: Literal["fw", "bw"] | None = None):
    batchsize = sampled_sequences.shape[0]
    if direction == "fw":
        sampled_sequences = torch.concat([FW_TOKEN.repeat(batchsize).unsqueeze(1), sampled_sequences[:, :-1]], dim=1)
    elif direction == "bw":
        sampled_sequences = torch.concat([sampled_sequences[:, :-1], BW_TOKEN.repeat(batchsize).unsqueeze(1)], dim=1)
    inputs, targets  = sampled_sequences[:, :-1], sampled_sequences[:, 1:] # reslice to get our input tokens and our shifted-by-1 targets
    if direction == "bw":
        inputs, targets = targets, inputs

    return inputs, targets


# Make loss function
loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)


##############################
#        Scheduling          #
##############################

# Infinite power law dicay is a simple power law learning rate schedule. seems to perform really well in practice as is simpler than OneCycle to tune.
# Does a linear warmup from a min_initial lr to the max_lr at the peak_step, then decays infinitely with a 1/x**(power_value)-type shape to it.
# These schedulers are multiplicative, that is why they scales from some base value to 1, which is what PyTorch's LambdaLR expects
infinite_power_law_decay    = lambda step, min_initial_mult, peak_step, exponent: min_initial_mult + step/peak_step * (1 - min_initial_mult) if step < peak_step else (step + 1. - peak_step) ** exponent
exp_decay_lr_scheduler_base = lambda step, decay: decay ** step

infinite_powah         = partial(infinite_power_law_decay, min_initial_mult=2e-2, peak_step=hyp['opt']['warmup_steps'], exponent=-.08)
infinite_powah_outputs = partial(infinite_power_law_decay, min_initial_mult=1.,   peak_step=0.,                         exponent=-.2)
pos_bias_decay_lr      = partial(exp_decay_lr_scheduler_base, decay=.995)

def init_param_groups_dict(net, base_lr):
    # the 'scheduler' attribute that we create here is not used by the optimizer, here we just use it to conveniently store all of these attributes.
    param_groups = {}

    # Multiply by our delta over the base lr-scaling curve
    scaled_lr = base_lr * hyp['opt']['lr_mult']['base']

    print("scaled lr:          ", "{:0.8f}".format(scaled_lr))

    # Decay is the default dictionary if there is no parameter name match
    param_groups['decay']                     = {'params': [], 'lr': scaled_lr,                                           'eps': 1e-9, 'betas': (.9,  .95), 'weight_decay': hyp['opt']['weight_decay'],  'scheduler': infinite_powah        }
    param_groups['position_bias_mult']        = {'params': [], 'lr': hyp['opt']['lr_mult']['position_bias']   *scaled_lr, 'eps': 1e-9, 'betas': (.9,  .95), 'weight_decay': 0,                           'scheduler': pos_bias_decay_lr     }
    param_groups['norm', 'bias', 'embedding'] = {'params': [], 'lr': hyp['opt']['lr_mult']['non_dot_products']*scaled_lr, 'eps': 1e-9, 'betas': (.9,  .95), 'weight_decay': 0,                           'scheduler': infinite_powah        }
    param_groups['output']                    = {'params': [], 'lr': hyp['opt']['lr_mult']['output_layer']    *scaled_lr, 'eps': 1e-9, 'betas': (.6,  .95), 'weight_decay': 0,                           'scheduler': infinite_powah_outputs}

    # Helper functions for matching parameters to dictionary keys
    in_list  = lambda name, keyword_list: any(keyword in name for keyword in keyword_list)
    to_tuple = lambda x: x if type(x) == tuple else (x,)

    # In order, search through the dictionary keys, and add to that dictionary if a value in the dictionary key matches the name.
    # 'decay' is the name of the default group, and is the only group with weight decay.
    for name, p in net.named_parameters():
        if p.requires_grad:
            target_param_dict = next(iter([k for k in param_groups.keys() if in_list(name, to_tuple(k))]), 'decay')
            param_groups[target_param_dict]['params'].append(p)

    return param_groups

def get_grad_norm(net):
    # Gets the entire grad norm of the network.
    grad_norm = torch.tensor(0., device=hyp['misc']['device'], dtype=torch.float64)
    for p in net.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.square()
    grad_norm = (grad_norm ** 0.5).item()
    return grad_norm


def grow_sequence_length(old_length, old_batchsize):
    # Dynamically grows the sequence length and changes the batchsize to avoid OOMs
    new_length        = min(2*old_length, hyp['misc']['sequence_length']['max'])
    new_batchsize     = tokens_per_batch_capacity // new_length

    print(f"| increasing sequence length (old: {old_length}, new: {new_length}), adjusting batchsize as necessary to fit (old: {old_batchsize}, new: {new_batchsize})")

    return new_length, new_batchsize


##############################
#          Logging           #
##############################

variables_to_log = ['epoch', 'curr_step', 'train_loss', 'val_loss_fw', 'val_loss_bw', 'pplx_fw', 'pplx_bw', 'train_acc', 'val_acc_fw', 'cum_secs']
# define the printing function and print the column heads
def print_training_details(columns_list, separator_left='  ', separator_right='  |', column_labels_only=False, is_final_entry=False):
    output_line = "|" # start with the left bar

    # Build the print string for the output:
    for column_entry in columns_list:
        output_line += separator_left + column_entry + separator_right

    if column_labels_only:
        print('-'*(len(output_line))) # print an initial upper dividing bar

    print(output_line)

    if column_labels_only or is_final_entry:
        print('-'*(len(output_line))) # print a lower divider bar

# The previous function was a shorter but slightly more heinous lambda, however, this may still cause you some pain. <3 :'(
def format_for_table(var_list, locals):
    int_format     = lambda x: f"{locals[x]}".rjust(len(x))
    default_format = lambda x: "{:0.4f}".format(locals[x]).rjust(len(x)) if len(f"{locals[x]:0.4f}") < 8 else "{:0.4f}".format(locals[x])[:7].rjust(len(x))
    blank_format   = lambda x: " "*len(x)

    out_list = [blank_format(v) if v not in locals else (int_format(v) if type(locals[v]) == int else default_format(v)) for v in var_list]
    return out_list


########################################
#           Train and Eval             #
########################################

def eval(net, use_extra_tokens=False):
    ####################
    # Evaluation  Mode #
    ####################

    # Do a slightly noisy fast eval over the max sequence length (should work okay as a rough general measurement of how we're doing)
    # Note that this is an approximation, it doesn't even necessarily use all of the requested tokens (but gets close because of the floor operation.)
    eval_batchsize           = max(math.floor(tokens_per_batch_capacity/(hyp['misc']['sequence_length']['max'])//16), 1) # Number of sequences per batch relative to the max-length batchsize capacity, downscale factor hardcoded to help prevent OOMs. Tunable
    num_eval_sequences       = hyp['opt']['num_eval_tokens']//hyp['misc']['sequence_length']['max']
    num_eval_steps           = num_eval_sequences//eval_batchsize

    # float32 here to prevent truncation errors
    val_loss_forward, val_acc_forward = torch.tensor(0., device=hyp['misc']['device'], dtype=torch.float), torch.tensor(0., device=hyp['misc']['device'], dtype=torch.float)
    val_loss_backward, val_acc_backward = torch.tensor(0., device=hyp['misc']['device'], dtype=torch.float), torch.tensor(0., device=hyp['misc']['device'], dtype=torch.float)
    global causal_mask, causal_mask_forward, causal_mask_backward
    global position_bias_base, position_bias_base_forward, position_bias_base_backward
    global causality

    with torch.no_grad():
        # Note: We eval at the maximum sequence length so that we can get an idea of how well the sequence length growing extrapolates out
        for _ in range(num_eval_steps):
            sampled_sequences = get_batch(data, key='eval', batchsize=eval_batchsize, length=hyp['misc']['sequence_length']['max'])


            causal_mask = causal_mask_forward
            position_bias_base = position_bias_base_forward
            causality = "forward"
            inputs, targets = split_batch(sampled_sequences, direction="fw" if use_extra_tokens else None)
            outputs = net(inputs)
            val_loss_forward += 1./num_eval_steps * loss_fn(outputs.flatten(0, 1).float(), targets.flatten(0, 1))
            val_acc_forward  += 1./num_eval_steps * (outputs.argmax(-1) == targets).float().mean()

            causal_mask = causal_mask_backward
            position_bias_base = position_bias_base_backward
            causality = "backward"
            inputs, targets = split_batch(sampled_sequences, direction="bw" if use_extra_tokens else None)
            outputs = net(inputs)
            val_loss_backward += 1./num_eval_steps * loss_fn(outputs.flatten(0, 1).float(), targets.flatten(0, 1))
            val_acc_backward  += 1./num_eval_steps * (outputs.argmax(-1) == targets).float().mean()
            
        val_perplexity_forward = 2.71828 ** val_loss_forward
        val_perplexity_backward = 2.71828 ** val_loss_backward

    return (
        val_acc_forward.item(), val_loss_forward.item(), val_perplexity_forward.item(),
        val_acc_backward.item(), val_loss_backward.item(), val_perplexity_backward.item(),
    )


def compute_loss(
        net: SpeedyLangNet,
        sampled_sequences: torch.Tensor,
        tokens_seen: int,
        mask: Literal["forward", "backward", "bidirectional"],
        curr_batchsize: int,
        curr_length: int,
        discrete_sampled_microbatch_steps: int,
        backward_prob: float = 1.0,
        use_extra_tokens: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, int]:
    """Returns outputs, loss, loss_fw, loss_bw, tokens_seen. Does *not* call `backward` on loss."""
    global causal_mask, causal_mask_backward, causal_mask_forward, causality
    global position_bias_base, position_bias_base_forward, position_bias_base_backward

    loss_fw, loss_bw = 0.0, 0.0

    inputs_fw, targets_fw = split_batch(sampled_sequences, direction="fw" if use_extra_tokens else None)
    inputs_bw, targets_bw = split_batch(sampled_sequences, direction="bw" if use_extra_tokens else None)
    if mask in ("forward", "bidirectional"):
        causal_mask = causal_mask_forward
        position_bias_base = position_bias_base_forward
        causality = "forward"
        outputs = net(inputs_fw)
        loss_fw = loss_fn(outputs.flatten(0, 1), targets_fw.flatten(0, 1))
        tokens_seen += curr_batchsize * curr_length
    if mask == "backward" or (mask == "bidirectional" and (torch.randint(0, 1000, (1,)).item() / 1000 <= backward_prob)):
        causal_mask = causal_mask_backward
        position_bias_base = position_bias_base_backward
        causality = "backward"
        outputs = net(inputs_bw)
        loss_bw = loss_fn(outputs.flatten(0, 1), targets_bw.flatten(0, 1))
        tokens_seen += curr_batchsize * curr_length
        
    loss = loss_fw + loss_bw
    loss = loss.div(discrete_sampled_microbatch_steps)
    return outputs, loss, inputs_fw, targets_fw, inputs_bw, targets_bw, float(loss_fw), float(loss_bw), tokens_seen
    


def train(
        mask: str, 
        num_tokens: int, 
        num_epochs: int, 
        num_steps: int,
        max_epochs_between_vals: float,
        backward_prob: float,
        adjust_backward_prob: bool,
        use_extra_tokens: bool,
        num_heads: int,
):

    #################
    #     Init      #
    #################
    # Full-run statistics variables
    cum_secs             = 0.  # cumulative seconds
    curr_microbatch_step = curr_step = 0
    tokens_seen          = 0
    distinct_tokens_seen = 0

    # Microbatch growing parameters
    # Leaving this hardcoded for now for simplicity, this helps keep the learning process stable.
    microbatch_steps = 0. # The noninteger estimate of microbatches required based upon the grad norm (sampled by dithering at each step.)
    discrete_sampled_microbatch_steps = max(1, int(microbatch_steps))

    # Start at the initial length and maximum allowable batchsize. The batchsize is adjusted so that we see roughly the same number of tokens per batch. This means that shorter sequence lengths will have much larger batch sizes.
    curr_length     = hyp['misc']['sequence_length']['initial']
    curr_batchsize  = tokens_per_batch_capacity // hyp['misc']['sequence_length']['initial']
    final_batchsize = tokens_per_batch_capacity /  hyp['misc']['sequence_length']['max']
    assert final_batchsize > 1, f"Error: Specified configuration takes up too much memory (calculated final batchsize {final_batchsize} is less than 1!)"

    # Validation parameters
    val_loss_fw, val_acc_fw, pplx_fw, val_loss_bw, val_acc_bw, pplx_bw = None, None, None, None, None, None

    train_losses, train_accs = [], []
    val_losses_forward, val_accs_forward, val_pplxs_forward = [], [], []
    val_losses_backward, val_accs_backward, val_pplxs_backward = [], [], []
    tokens_seen_list_train, epoch_list_train, steps_list_train = [], [], []
    epoch_by_distinct_tokens_seen_list_train = []
    tokens_seen_list_val, epoch_list_val, steps_list_val = [], [], []
    cumulative_times_val, epoch_by_distinct_tokens_seen_list_val = [], []
    backward_probs = []

    # Get network
    net = make_net(num_heads=num_heads)

    # Get the total number of parameters in our model and use that to generate/calculate the base lr.
    total_trainable_params = sum([p.data.numel() if p.requires_grad else 0 for p in net.parameters()])

    print('-'*(40))
    print(f"total trainable params: {total_trainable_params:,}")
    print('-'*(40))

    # Briefly log some details up front. (TODO: Condense nicely later.)
    print("curr_batchsize:     ", curr_batchsize)
    print("final_batchsize:    ", tokens_per_batch_capacity // hyp['misc']['sequence_length']['max'])
    print("max_sequence_length:", max_sequence_length)


    #####################
    # Scaling Equations #
    #####################

    # These equations are a result of rough general exponential/power law fits between parameters that worked for the 46M and 1.5B run
    # They seem to transfer not too badly when interpolating, however, they're far from perfect and assume 40 GB of memory (so if you use)
    # a smaller card, you might struggle a bit here. All in all -- this is still in alpha, but seems to be very useful within a limited arena
    # of making arbitrary models between 45M and 1.5B

    # A very, very pared down version of the gpt-3 training lr scaling rule roughly fit. It's used as a loose general base for the run LRs.
    base_lr = 9e7 / math.log(total_trainable_params)**8.8

    # The base value that we raise to the value of our loss in order to determine how much weight decay we need (exponentially strong as we approach 0.)
    weight_decay_pow_base = .007 * ((.01 * math.log(total_trainable_params))) ** (-4)

    # This defines how quickly we expect grad_norm drops for microbatch scheduling -- slightly faster for smaller models, slightly slower for larger models
    # Note: This will interact with really aggressive weight decay, some training runs may slow down a lot near the end as a result.
    microbatch_expected_grad_norm_pow = -.677 * math.log(total_trainable_params) ** -.2

    # Bit of a strange approximation, but this seemed
    microbatch_grad_norm_steps_scale = math.log(total_trainable_params) * total_trainable_params

    # Create multiple parameter groups based on parameter name, as certain kinds of parameters seem to work best
    # with specific combinations of learning rates and schedulers
    param_groups_dict = init_param_groups_dict(net, base_lr)
    opt               = torch.optim.AdamW(param_groups_dict.values(), fused=True)
    scheduler         = torch.optim.lr_scheduler.LambdaLR(opt, [k['scheduler'] for k in param_groups_dict.values()])


    #################
    # Training Mode #
    #################

    ## print out the training column headers before each run.
    print_training_details(variables_to_log, column_labels_only=True)

    ## For accurately timing GPU code
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize() ## clean up any pre-net setup operations
    starter.record()

    net.train()
    global causal_mask_forward, causal_mask_backward, causal_mask
    global position_bias_base_forward, position_bias_base_backward, position_bias_base
    global causality

    epoch = 0.0

    # Main loop. Most of the complexity here is in the dynamic growing scheduler(s).
    while curr_step < num_steps and tokens_seen < num_tokens and epoch < num_epochs:
        sampled_sequences = get_batch(data, key='train', batchsize=curr_batchsize, length=curr_length)

        outputs, loss, inputs_fw, targets_fw, inputs_bw, targets_bw, loss_fw, loss_bw, tokens_seen = compute_loss(
            net=net,
            sampled_sequences=sampled_sequences,
            tokens_seen=tokens_seen,
            mask=mask,
            curr_batchsize=curr_batchsize,
            curr_length=curr_length,
            discrete_sampled_microbatch_steps=discrete_sampled_microbatch_steps,
            backward_prob=backward_prob,
            use_extra_tokens=use_extra_tokens,
        )
        loss.backward()
        backward_probs.append(backward_prob)
        if adjust_backward_prob and loss_bw != 0.0:  # only adjust if we've just done a backward pass
            pplx_fw, pplx_bw = 2.71828 ** loss_fw, 2.71828 ** loss_bw
            backward_prob = max(min(1.0, pplx_bw / pplx_fw), 0.01)  # if backward is easier, do it less frequently (but not never)

        epoch = tokens_seen/len(data['train'])
        distinct_tokens_seen += len(inputs_fw)

        do_stop = (curr_step >= num_steps) or (tokens_seen >= num_tokens) or (epoch >= num_epochs)
        do_eval = (
            do_stop 
            or (
                (curr_microbatch_step % discrete_sampled_microbatch_steps == 0)
                and (
                    (curr_step % hyp['opt']['eval_every'] == 0 and curr_step > 0) 
                    or (epoch_list_val and (epoch - epoch_list_val[-1] >= max_epochs_between_vals))
                )
            )
        )

        # Quick non-eval summary every N training steps, at the end of every microbatch group
        if do_eval or (curr_step % 10 == 0 and curr_microbatch_step % discrete_sampled_microbatch_steps == 0):
            train_acc          = (outputs.detach().argmax(-1) == targets_fw).float().mean().item()
            train_loss         = loss.detach().cpu().item()

            epoch_list_train.append(epoch)
            epoch_by_distinct_tokens_seen_list_train.append(distinct_tokens_seen/len(data['train']))
            tokens_seen_list_train.append(tokens_seen)
            steps_list_train.append(curr_step)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # don't print training details if we're about to do a full eval
            if not do_eval:
                train_summary_vars = {'epoch': epoch, 'curr_step': curr_step, 'train_loss': train_loss, 'train_acc': train_acc}
                print_training_details(format_for_table(variables_to_log, locals=train_summary_vars))


        # Once we've accumulated steps over all of our microbatches, take a single full-batchsize step.
        if curr_microbatch_step % discrete_sampled_microbatch_steps == 0:
            # Step the optimizer, then scheduler
            opt.step()

            # Dynamic weight decay scheduling. Based upon something similar to the reciprocal of the perplexity of the network over the data [inspired by section 5 of https://arxiv.org/pdf/2204.02311.pdf]
            # Smaller models have a higher base, and weight decay kicks in more sharply later. For larger models, it activates more early
            opt.param_groups[0]['weight_decay'] = 1./weight_decay_pow_base**(loss.detach()+1e-8).item() * hyp['opt']['weight_decay']
            scheduler.step()

            # Check if we need to double our sequence length
            if curr_step % hyp['misc']['sequence_length']['growth_steps'] == 0 and curr_step != 0 and curr_length < hyp['misc']['sequence_length']['max']:
                curr_length, curr_batchsize = grow_sequence_length(curr_length, curr_batchsize)

            # The next several lines calculate a dynamic batchsize, simulated through manual dithering
            # There could be improvements or losses in changing the dithering strategy, since determinism and gradient descent can lead to some very not-so-nice (and subtle) loss oscillations.
            if curr_step % hyp['opt']['microbatch']['sample_every'] == 0:
                grad_norm = get_grad_norm(net)

                grad_norm_per_param = grad_norm/(total_trainable_params**.5) # This should keep the expected grad norm per parameter roughly the same (ignoring initializations) unless I did my napkin math wrong (feel free to correct it and test it out if so! <3 :') )
                grad_norm_target    = (((microbatch_grad_norm_steps_scale * (curr_step + 1e-2))) ** microbatch_expected_grad_norm_pow)
                ratio_diff          = grad_norm_per_param/(grad_norm_target)

                # Update the fractional number of steps based on the % difference between the grad norm and expected grad norm.
                microbatch_steps *= 1. + (hyp['opt']['microbatch']['sample_every'] * hyp['opt']['microbatch']['scale_lr'] * (ratio_diff - 1))
                microbatch_steps  = max(microbatch_steps, 1e-1) # Clamp to keep this from going to zero, so that we can bounce back if needed

            # simple bernoulli dithering with probabilities based on how close we are to each integer
            base, dither_prob = divmod(microbatch_steps, 1)

            # Randomly sample next accumulate steps to use. This is the dithered operation, the 'microbatch_steps' is the noninteger accumulator between steps.
            discrete_sampled_microbatch_steps = max(1, int(base + torch.bernoulli(torch.tensor(dither_prob)).item())) # bernoulli via torch to save an unnecesary import :)

            opt.zero_grad()

            # reset microbatch steps and increment current step
            curr_microbatch_step = 0
            curr_step += 1

            # Since we're not running over epochs anymore, we have to manually calculate roughly what epoch it is. This is different than the standard random derangement of sampled sequences and has different pros/cons, is my understanding. :thumbsup:
            epoch = tokens_seen/len(data['train'])

        if do_eval:
            ender.record()
            torch.cuda.synchronize()

            cum_secs += 1e-3 * starter.elapsed_time(ender)
            train_loss = loss.detach().cpu().item() # Update the loss for the training details printout

            net.eval()
            val_acc_fw, val_loss_fw, pplx_fw, val_acc_bw, val_loss_bw, pplx_bw = eval(net, use_extra_tokens=use_extra_tokens)
            
            val_losses_forward.append(val_loss_fw)
            val_accs_forward.append(val_acc_fw)
            val_pplxs_forward.append(pplx_fw)
            val_losses_backward.append(val_loss_bw)
            val_accs_backward.append(val_acc_bw)
            val_pplxs_backward.append(pplx_bw)
            steps_list_val.append(curr_step)
            tokens_seen_list_val.append(tokens_seen)
            epoch_list_val.append(epoch)
            epoch_by_distinct_tokens_seen_list_val.append(distinct_tokens_seen/len(data['train']))
            cumulative_times_val.append(cum_secs)

            # Print out our training details
            ## We also check to see if we're on our final eval loop (assum that max_curr_step lines up with the eval_every value) so we can print the 'bottom' of the table for each round.
            is_final_eval = (curr_step >= num_steps) # If we're at the end of training, add a line after the end of the run
            print_training_details(format_for_table(variables_to_log, locals=locals()), is_final_entry=is_final_eval)

            torch.cuda.synchronize()
            starter.record()
            net.train()

        if do_stop:
            break
    
        curr_microbatch_step += 1

    return (
        net,
        val_loss_fw, val_loss_bw, total_trainable_params, train_losses, val_losses_forward,
        val_losses_backward, val_accs_backward, val_pplxs_backward,
        train_accs, val_accs_forward, val_pplxs_forward,
        tokens_seen_list_train, epoch_list_train, steps_list_train,
        epoch_by_distinct_tokens_seen_list_train,
        tokens_seen_list_val, epoch_list_val, steps_list_val,
        cumulative_times_val, epoch_by_distinct_tokens_seen_list_val,
        backward_probs,
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT-like model on the wikitext-103 dataset.")

    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("--savefile", type=str, default="results.csv")
    parser.add_argument("--append", action="store_true")

    parser.add_argument("--gpu_capacity_scalar", type=float, default=1.0, help="The max token capacity is multiplied by this.")

    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--model_scale", type=float, default=1.0, nargs="+", help="The scale of the model to train.")
    parser.add_argument("--scale_manually", action="store_true", help="If set, depth and width will be set through the cli.")
    parser.add_argument("--depth", type=int, default=8, nargs="+", help="The depth of the model to train.")
    parser.add_argument("--width", type=int, default=384, nargs="+", help="The width of the model to train.")
    parser.add_argument("--num_heads", type=int, default=1, nargs="+", help="The number of heads to use in the attention mechanism.")
    parser.add_argument("--mask", type=str, choices=["forward", "backward", "bidirectional"], nargs="+", default="forward", help="The type of mask to use for the attention mechanism.")
    parser.add_argument("--backward_prob", type=float, default=1.0, nargs="+", help="Only relevant if mask==bidirectional")
    parser.add_argument("--adjust_backward_prob", type=int, nargs="+", default=0, help="If set, backward prob will be dynamically adjusted.")
    parser.add_argument("--use_extra_tokens", action="store_true", help="If set, the model will use extra tokens for the forward and backward pass.")

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=int(1e6))
    parser.add_argument("--num_tokens", type=int, default=int(1e12))
    parser.add_argument("--max_epochs_between_vals", type=float, default=0.25)


    args = parser.parse_args()
    args.mask = args.mask if isinstance(args.mask, list) else [args.mask]
    args.model_scale = args.model_scale if isinstance(args.model_scale, list) else [args.model_scale]
    args.depth = args.depth if isinstance(args.depth, list) else [args.depth]
    args.width = args.width if isinstance(args.width, list) else [args.width]
    args.num_heads = args.num_heads if isinstance(args.num_heads, list) else [args.num_heads]
    args.backward_prob = args.backward_prob if isinstance(args.backward_prob, list) else [args.backward_prob]
    args.adjust_backward_prob = args.adjust_backward_prob if isinstance(args.adjust_backward_prob, list) else [args.adjust_backward_prob]

    args.backward_prob = list(set(max(0.0, min(1.0, p)) for p in args.backward_prob))
    args.adjust_backward_prob = list(set([bool(x) for x in args.adjust_backward_prob]))

    return args


def get_settings(args: argparse.Namespace) -> list:
    if args.scale_manually:
        standard_settings = list(itertools.product(args.depth, args.width, args.num_heads, args.mask))
        standard_settings = [
            (1.0, depth, width, num_heads, mask) 
            for depth, width, num_heads, mask in standard_settings
            if width % num_heads == 0
        ]
    else:
        standard_settings = list(itertools.product(args.model_scale, args.num_heads, args.mask))
        standard_settings = [(ms, None, None, num_heads, mask) for ms, num_heads, mask in standard_settings]

    bw_settings = [(bp, False) for bp in args.backward_prob if False in args.adjust_backward_prob]
    if True in args.adjust_backward_prob:
        bw_settings.append((1.0, True))

    combined_settings = [
        (ms, depth, width, num_heads, mask, bp, adjust_backward_prob)
        for ms, depth, width, num_heads, mask in standard_settings
        for bp, adjust_backward_prob in bw_settings
        if mask == "bidirectional"
    ]
    combined_settings.extend([(ms, depth, width, num_heads, mask, 0.0, False) for ms, depth, width, num_heads, mask in standard_settings if mask != "bidirectional"])
    
    return combined_settings


def main() -> None:
    args = get_args()
    settings = get_settings(args)
    global_run_num = 0
    total_num_runs = int(len(settings) * args.num_runs)
    global hyp, model_scale
    change_gpu_token_capacity(args.gpu_capacity_scalar)

    for setting_num, (model_scalar, depth, width, num_heads, mask, backward_prob, adjust_backward_prob) in enumerate(settings):
        seed = args.seed
        for run in range(args.num_runs):
            change_model_scale(model_scalar, depth, width)  # has to run here; remove_layers influences hyp
            depth = hyp['net']['num_blocks']  # remove_layers also influences depth in hyp; but I want to store the original depth. Need to get it here in case I changed it through model_scalar instead of depth
            global_run_num += 1
            title = (
                f"::: STARTING RUN {global_run_num}/{total_num_runs} "
                f"(Setting {setting_num+1}/{len(settings)}, Run {run+1}/{args.num_runs})\n"
                f":::    {mask=}\n:::    {model_scale=:.4f}\n"
                f":::    {depth=}\n:::    {width=}\n"
                f":::    {backward_prob=}\n:::    {adjust_backward_prob=}"
            )
            max_len = max(len(line) for line in title.split("\n"))
            title = "\n".join([line + " " * (max_len - len(line)) + " :::" for line in title.split("\n")])
            sep = ":" * max(len(line) for line in title.split("\n"))
            title = "\n\n" + "\n".join([sep, title, sep]) + "\n\n"
            print(title)

            torch.manual_seed(seed)
            (
                net,
                val_loss_fw, val_loss_bw, num_params, train_losses, val_losses_forward,
                val_losses_backward, val_accs_backward, val_pplxs_backward,
                train_accs, val_accs_forward, val_pplxs_forward,
                tokens_seen_train, epochs_train, steps_train,
                epoch_by_distinct_tokens_seen_train,
                tokens_seen_val, epochs_val, steps_val,
                cumulative_times_val, epoch_by_distinct_tokens_seen_val,
                backward_probs,
            ) = train(
                mask=mask,
                num_tokens=args.num_tokens,
                num_epochs=args.num_epochs,
                num_steps=args.num_steps,
                max_epochs_between_vals=args.max_epochs_between_vals,
                backward_prob=backward_prob,
                adjust_backward_prob=adjust_backward_prob,
                use_extra_tokens=args.use_extra_tokens,
                num_heads=num_heads,
            )

            cut_accs_fw, cut_losses_fw, cut_pplxs_fw, cut_accs_bw, cut_losses_bw, cut_pplxs_bw, n_layers_removed = [], [], [], [], [], [], []
            for n in range(1, hyp['net']['num_blocks']):
                print(f"Evaluating model with layers removed: {n}")
                net = remove_layers(net, 1)  # reduce the depth one by one
                acc_fw, loss_fw, pplx_fw, acc_bw, loss_bw, pplx_bw = eval(net, use_extra_tokens=args.use_extra_tokens)
                cut_accs_fw.append(acc_fw)
                cut_losses_fw.append(loss_fw)
                cut_pplxs_fw.append(pplx_fw)
                cut_accs_bw.append(acc_bw)
                cut_losses_bw.append(loss_bw)
                cut_pplxs_bw.append(pplx_bw)
                n_layers_removed.append(n)
                print(f"{n=:.2f}, {acc_fw=:.2f}, {loss_fw=:.2f}, {pplx_fw=:.2f}, {acc_bw=:.2f}, {loss_bw=:.2f}, {pplx_bw=:.2f}\n")

            del net

            results = {
                "final_val_loss_fw": [val_loss_fw],
                "final_val_loss_bw": [val_loss_bw],
                "mask": [mask],
                "model_scale": [model_scale],
                "depth": depth,
                "width": [hyp['net']['residual_depth']],
                "num_heads": [num_heads],
                "num_params": [num_params],
                "run_num": [run],
                "seed": [seed],
                "adjust_backward_prob": [adjust_backward_prob],
                "use_extra_tokens": [args.use_extra_tokens],
                "initial_backward_prob": [backward_prob],
                "backward_probs": [str(backward_probs)],
                "train_losses": [str(train_losses)],
                "train_accs": [str(train_accs)],
                "val_losses_fw": [str(val_losses_forward)],
                "val_accs_fw": [str(val_accs_forward)],
                "val_pplxs_fw": [str(val_pplxs_forward)],
                "val_losses_bw": [str(val_losses_backward)],
                "val_accs_bw": [str(val_accs_backward)],
                "val_pplxs_bw": [str(val_pplxs_backward)],
                "tokens_seen_train": [str(tokens_seen_train)],
                "epochs_train": [str(epochs_train)],
                "steps_train": [str(steps_train)],
                "epoch_by_distinct_tokens_seen_train": [str(epoch_by_distinct_tokens_seen_train)],
                "tokens_seen_val": [str(tokens_seen_val)],
                "epochs_val": [str(epochs_val)],
                "steps_val": [str(steps_val)],
                "cumulative_times_val": [str(cumulative_times_val)],
                "epoch_by_distinct_tokens_seen_val": [str(epoch_by_distinct_tokens_seen_val)],
                "n_layers_removed": [str(n_layers_removed)],
                "cut_accs_fw": [str(cut_accs_fw)],
                "cut_losses_fw": [str(cut_losses_fw)],
                "cut_pplxs_fw": [str(cut_pplxs_fw)],
                "cut_accs_bw": [str(cut_accs_bw)],
                "cut_losses_bw": [str(cut_losses_bw)],
                "cut_pplxs_bw": [str(cut_pplxs_bw)],
            }

            df = pl.DataFrame(results)

            if args.save:
                if not os.path.exists(args.savefile) or ((setting_num == run == 0) and not args.append):
                    df.write_csv(args.savefile)
                else:
                    with open(args.savefile, "ab") as f:
                        df.write_csv(f, include_header=False)

            seed += 1

if __name__ == "__main__":
    main()
