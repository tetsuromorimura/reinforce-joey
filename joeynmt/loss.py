# coding: utf-8
"""
Module to implement training loss
"""

import torch
from torch import nn, Tensor
from torch.autograd import Variable
from joeynmt.metrics import bleu
import sacrebleu
import numpy as np
from random import sample
from joeynmt.model import RewardRegressionModel


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                        reduction='sum')
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction='sum')

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0-self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index,
            as_tuple=False)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1),
                vocab_size=log_probs.size(-1))
            # targets: distributions with batch*seq_len x vocab_size
            assert log_probs.contiguous().view(-1, log_probs.size(-1)).shape \
                == targets.shape
        else:
            # targets: indices with batch*seq_len
            targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets)
        return loss

class ReinforceLoss(nn.Module):
    """
    Reinforce Loss
    """
    def __init__(self, baseline, use_cuda, reward: str="bleu"):
        super(ReinforceLoss, self).__init__()
        self.baseline = baseline
        self.reward = reward
        self.bleu = []
        self.counter = 0 
        self.use_cuda = use_cuda
        self.critic_loss = torch.nn.MSELoss(reduction='sum')
        # hyperparams for learned reward: TODO move to config
        if self.baseline == "learned_reward_baseline":
            self.all_targets_and_outputs = torch.FloatTensor()
            self.all_bleus = torch.FloatTensor()
            #hyperparameter for learned reward baseline
            self.learned_baseline_loss = torch.nn.MSELoss(reduction='sum')
            self.padding_size = 180
            self.hidden_size = 200
            self.steps = 500 # steps should be more than batch size
            self.max_elements = 2000 # maximum number of samples to train
            self.max_training_size = 10000
            self.learned_baseline_model = RewardRegressionModel(2*self.padding_size, self.hidden_size, 1)
            self.learned_baseline_learning_rate = 1e-4
            self.learned_baseline_optimizer = torch.optim.Adam(self.learned_baseline_model.parameters(), lr=self.learned_baseline_learning_rate)
            if self.use_cuda:
                self.critic_loss.cuda()
                self.learned_baseline_model.cuda()
                self.learned_baseline_loss.cuda()
                self.all_targets_and_outputs = torch.FloatTensor().cuda()
                self.all_bleus = torch.FloatTensor().cuda()

    def forward(self, predicted, gold, log_probs, stacked_output, targets):

        bleu_scores = [bleu([prediction], [gold_ref]) \
                for prediction, gold_ref in zip(predicted, gold)]
        # save unscaled rewards for logging
        unscaled_rewards = bleu_scores
        if self.reward == "constant":
            loss = sum([log_prob for log_prob in log_probs])

        elif self.reward == "scaled_bleu":
                def scale(reward, a, b, minim, maxim):
                    if maxim-minim == 0:
                        return 0
                    else: 
                        return (((b-a)*(reward - minim))/(maxim-minim)) + a 

                # scale locally
                maxim = max(bleu_scores)
                minim = min(bleu_scores)
                bleu_scores = [scale(score, -0.5, 0.5, minim, maxim) for score in bleu_scores]

        elif self.reward == "bleu":
            if self.baseline == "average_reward_baseline":
                # this baseline is calculated by using a global average
                self.bleu.append(sum(bleu_scores))
                self.counter += len(bleu_scores)
                average_bleu = sum([score for score in self.bleu])/self.counter
                bleu_scores = [score - average_bleu for score in bleu_scores]
            
            elif self.baseline == "learned_reward_baseline": 
                # TODO currently not working as intended 
                with torch.enable_grad(): 
                    stacked_output = stacked_output.float()
                    # pad target tensor with zeros
                    if self.use_cuda:
                        padded_targets = torch.zeros(list(targets.size())[0], self.padding_size, dtype=torch.float, device='cuda')
                        padded_outputs = torch.zeros(list(targets.size())[0], self.padding_size, dtype=torch.float, device='cuda')
                    else:
                        padded_targets = torch.zeros(list(targets.size())[0], self.padding_size, dtype=torch.float)
                        padded_outputs = torch.zeros(list(targets.size())[0], self.padding_size, dtype=torch.float)
                    padded_targets[:, :list(targets.size())[1]] = targets.float()
                    padded_outputs[:, :list(stacked_output.size())[1]] = stacked_output.float()
                    stacked_training_data = torch.cat([padded_outputs, padded_targets], 1)           
                    stacked_training_data.cuda()    
                    bleu_tensor = torch.FloatTensor(bleu_scores).cuda()
                    bleu_tensor = bleu_tensor.unsqueeze(1)
                    if self.use_cuda:
                        bleu_tensor.cuda()
                    N = list(self.all_targets_and_outputs.size())[0]
                    if N != 0: 
                        x = self.all_targets_and_outputs
                        y = self.all_bleus
                        if N > self.max_elements: # select N samples
                            indices = sample(range(N),self.max_elements)
                            x = torch.from_numpy(x.cpu().numpy()[indices, :]).cuda()
                            y = torch.from_numpy(y.cpu().numpy()[indices, :]).cuda()
                        if self.use_cuda:
                            x.cuda()
                            y.cuda()
                        x_test = stacked_training_data
                        for t in range(self.steps):
                            y_pred = self.learned_baseline_model(x)
                            loss = self.learned_baseline_loss(y_pred, y)
                            self.learned_baseline_optimizer.zero_grad()
                            loss.backward()
                            self.learned_baseline_optimizer.step()
                        learned_bleus = self.learned_baseline_model(x_test).squeeze(1).tolist()
                        new_bleus = [score - learned_bleu for score, learned_bleu in zip(bleu_scores, learned_bleus)]
                        bleu_scores = new_bleus
                # remove training data if its too much 
                if self.all_targets_and_outputs.size()[0] > self.max_training_size:
                    rows_to_keep = self.all_targets_and_outputs.size()[0] - stacked_training_data.size()[0]
                    indices = sample(range(N),rows_to_keep)
                    self.all_targets_and_outputs = self.all_targets_and_outputs[indices,:]
                    self.all_bleus = self.all_bleus[indices,:]
                    if self.use_cuda:
                        self.all_targets_and_outputs.cuda()
                        self.all_bleus.cuda()
                # append current training data
                self.all_bleus = torch.cat([self.all_bleus, bleu_tensor], 0)
                self.all_targets_and_outputs = torch.cat([self.all_targets_and_outputs, stacked_training_data], 0)

            # calculate PG loss with rewards and log probs
            loss = sum([log_prob*bleu_score \
                for log_prob, bleu_score in zip(log_probs, bleu_scores)])
        return loss, bleu_scores, unscaled_rewards
