---
layout: post
title: Chinese Word Segmentation
description: 中文分词算法--基于词表的分词方法
categories:
    - NLP
comments: true
permalink: 
---

这遍文章介绍NLP中的中文分词方法--基于词表的分词方法。来源于宗成庆老师的PPT。

## 最大匹配法(Maximum Matching)

* 正向最大匹配算法(Forward MM, FMM)
* 逆向最大匹配算法(Backward MM, BMM)
* 双向最大匹配算法(Bi-directional MM)

## 问题：给定一个句子
This is a \emph{LaTeX} file.
