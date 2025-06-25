import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from torchvision import transforms


model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda', is_eval=True)
vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])


def nms(moments, scores, pre_mom, pre_score, thresh):
    scores = scores + pre_score * 0.0
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    pre_mom = pre_mom[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True
    return moments[~suppressed], pre_mom[~suppressed], scores[~suppressed]


def iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0].float(), gt[1].float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union


def get_dynamic_scores(scores, stride, masks, ths=0.0005, sigma=1):
    def gaussian_kernel(size, sigma=1):
        size = int(size) // 2
        x = np.arange(-size, size+1)
        normal = 1 / (np.sqrt(2.0 * np.pi) * sigma)
        g =  np.exp(-x**2 / (2.0 * sigma**2)) * normal
        return g
    
    def nchk(f, f1, f2, ths):
        return (((3 * f) > ths) | ((2 * f + f1) > ths) | ((f + f1 + f2) > ths))
    
    gstride = min(stride - 2, 3)
    if (stride < 3):
        gkernel = torch.ones((1, 1, 1)).to('cuda')
    else:
        gkernel = gaussian_kernel(gstride, sigma)
        gkernel = torch.from_numpy(gkernel).float().to('cuda')
        gkernel = gkernel.view(1, 1, -1)
    gscore = F.conv1d(scores.view(-1, 1, scores.size(-1)), gkernel).view(scores.size(0), -1)

    diffres = torch.diff(gscore).to('cuda')
    pad_left = torch.zeros((diffres.size(0), (masks.size(-1) - diffres.size(-1)) // 2)).to('cuda')
    pad_right = torch.zeros((diffres.size(0), masks.size(-1) - diffres.size(-1) - pad_left.size(-1))).to('cuda')
    diffres = torch.cat((pad_left, diffres, pad_right), dim = -1) * masks

    dynamic_scores = np.zeros((diffres.size(0), diffres.size(-1)))
    dynamic_idxs = np.zeros((diffres.size(0), diffres.size(-1)))

    for idx in range(diffres.size(0)):
        f1 = f2 = f3 = 0
        d_score = 0
        d_idx = 0
        for i in range(diffres.size(-1)):
            f3 = f2
            f2 = f1
            f1 = diffres[idx][i]
            if nchk(f1, f2, f3, ths):
                d_score += max(3 * f1,2 * f1 + f2,f1 + f2 + f3)
            else:
                d_idx = i
                d_score = 0

            dynamic_idxs[idx][i] = d_idx / scores.size(-1)
            dynamic_scores[idx][i] = d_score

    dynamic_idxs = torch.from_numpy(dynamic_idxs).to('cuda')
    dynamic_scores = torch.from_numpy(dynamic_scores).to('cuda')

    return dynamic_idxs, dynamic_scores

def fliter_sentences(centroid_text, sentences, topK = 3):
    '''

    :param centroid_text:
    :param sentences:
    :param topK:
    :return:
    '''
    with torch.no_grad():
        centroid_text_ = model.tokenizer(centroid_text, padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to('cuda')
        centroid_text_output = model.Qformer.bert(centroid_text_.input_ids, attention_mask=centroid_text_.attention_mask, return_dict=True)
        centroid_text_feat = model.text_proj(centroid_text_output.last_hidden_state[:,0,:])
        centroid_text_feat = F.normalize(centroid_text_feat, dim=-1)
        v1 = centroid_text_feat

        text_feats = []
        words_ = sentences[0].split(' ')
        words = []
        for w in words_:
            if len(w)>2:
                words.append(w)
        for i in words:
            # for i in range(2): # 单词和短语
            # sentence = i
            text = model.tokenizer(i, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
                'cuda')
            text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
            text_feat = model.text_proj(text_output.last_hidden_state[:,0,:])
            text_feats.append(F.normalize(text_feat, dim=-1))

        v2 = torch.cat(text_feats)

        # 计算相似度
        similarities = torch.mm(v1, v2.T)  # shape: [1, 12]

        # 提取前三个索引
        topk_values, topk_indices = similarities.topk(min(topK, len(words)), dim=1)
        topk_indices = topk_indices.squeeze().cpu().tolist()  # 转换为 CPU 上的列表

        if len(words)==1:
            topk_indices = [topk_indices]
        # 对topk_indices应保持词序，然后获取对应的句子,
        topk_indices.sort()
        top_words = [words[idx] for idx in topk_indices]
    return [(' ').join(top_words)], topk_values.cpu()

def calc_scores(video_features, sentences):
    with torch.no_grad():
        text = model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')                    
        text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feat = model.text_proj(text_output.last_hidden_state[:,0,:])
    
    v1 = F.normalize(text_feat, dim=-1)
    v2 = F.normalize(torch.tensor(video_features, device='cuda', dtype=v1.dtype), dim=-1)
    scores = torch.einsum('md,npd->mnp', v1, v2)
    scores, _ = scores.max(dim=-1)
    scores = scores.mean(dim=0, keepdim=True)

    return scores


def generate_proposal(video_features, sentences, stride, max_stride, nms_thresh=0.3):
    scores = calc_scores(video_features, sentences)
    masks = (scores > 0.2).float()
    scores = scores * masks
    stride = min(stride, scores.size(-1)//2)
    dynamic_idxs, dynamic_scores = get_dynamic_scores(scores, stride, masks)

    # static scores
    flattened_proposals = []
    flattened_scores = []
    flattened_prefix = []
    flattened_prescore = []
    for kernel_size in range(stride, min(scores.size(-1)+1, max_stride+1), stride):
        kernel = torch.ones((1, 1, kernel_size)).to('cuda')
        inner_sum = F.conv1d(scores.view(-1, 1, scores.size(-1)), kernel).view(scores.size(0), -1)
        inner_num = F.conv1d(masks.view(-1, 1, masks.size(-1)), kernel).view(masks.size(0), -1)
        outer_sum = (scores * masks).sum(dim=-1, keepdim=True) - inner_sum
        outer_num = masks.sum(dim=-1, keepdim=True) - inner_num
        static_scores = inner_sum / kernel_size - outer_sum / outer_num
        proposals = torch.arange(0, static_scores.size(-1)).to('cuda')
        proposals = torch.stack([proposals, proposals + kernel_size], dim=-1) / scores.size(-1)

        dynamic_idxs_tmp = dynamic_idxs.narrow(-1, 0, static_scores.size(-1))
        dynamic_scores_tmp = dynamic_scores.narrow(-1, 0, static_scores.size(-1))
        for idx in range(static_scores.size(0)):
            mask = static_scores[idx] > -1e3
            if idx >= len(flattened_proposals):
                flattened_proposals.append(proposals[mask])
                flattened_scores.append(static_scores[idx][mask])
                flattened_prefix.append(dynamic_idxs_tmp[idx][mask])
                flattened_prescore.append(dynamic_scores_tmp[idx][mask])
            else:
                flattened_proposals[idx] = torch.concat([flattened_proposals[idx], proposals[mask]], dim=0)
                flattened_scores[idx] = torch.concat([flattened_scores[idx], static_scores[idx][mask]], dim=0)
                flattened_prefix[idx] = torch.concat([flattened_prefix[idx], dynamic_idxs_tmp[idx][mask]], dim=0)
                flattened_prescore[idx] = torch.concat([flattened_prescore[idx], dynamic_scores_tmp[idx][mask]], dim=0)

    # NMS
    filtered_proposals = []
    filtered_scores = []
    filtered_prefix = []
    for idx in range(len(flattened_proposals)):
        if len(flattened_proposals[idx]) > 0:
            nms_proposals, nms_prefix, nms_scores = nms(flattened_proposals[idx], flattened_scores[idx], flattened_prefix[idx], flattened_scores[idx], nms_thresh)
            filtered_proposals.append(nms_proposals)
            filtered_scores.append(nms_scores)
            filtered_prefix.append(nms_prefix)
        else:
            filtered_proposals.append([])
            filtered_scores.append([])
            filtered_prefix.append([])

    return filtered_proposals, filtered_scores, filtered_prefix, scores


def localize(video_feature, duration, query_json, stride, max_stride, purif=False):
    '''
    是否可以逐词进行单独的检索？然后再聚合？
    现在方案是先把相关词汇聚合然后，一次检索
    :param video_feature:
    :param duration:
    :param query_json:
    :param stride:
    :param max_stride:
    :param purif:
    :return:
    '''
    answer = []
    purif_words = []
    purif_scores = []
    for query in query_json:
        if purif:
            query['descriptions'], words_scores = fliter_sentences('violence, criminal', query['descriptions'], topK=3)
            # query['descriptions'], words_scores = 'violence, criminal', [1.,1.]
            purif_words.append(query['descriptions'])
            purif_scores.append(words_scores)
        proposals, scores, pre_proposals, ori_scores = generate_proposal(video_feature, query['descriptions'], stride, max_stride)

        if len(proposals[0]) == 0:
            static_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            dynamic_pred = np.array([0.0, 0.0, 0.0])
            scores = np.array([1.0, 1.0, 1.0])
        else:
            static_pred = proposals[0][:10] * duration
            dynamic_pred = pre_proposals[0][:10] * duration
            scores = scores[0][:10]
            scores = scores / scores.max()
        query['response'] = []
        for i in range(len(static_pred)):
            query['response'].append({
                'start': float(dynamic_pred[i]),
                'static_start': float(static_pred[i][0]),
                'end': float(static_pred[i][1]),
                'confidence': float(scores[i])
            })
        answer.append(query)
    if purif:
        return answer, purif_words, purif_scores
    return answer, purif_words, purif_scores

