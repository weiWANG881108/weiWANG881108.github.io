---
layout: post
title: Chinese Word Segmentation
description: 中文分词算法--基于词表的分词方法
categories:
    - NLP
comments: true
math: true
permalink: 
---

这遍文章介绍NLP中的中文分词方法--基于词表的分词方法。来源于宗成庆老师的PPT。

## 最大匹配法(Maximum Matching)

* 正向最大匹配算法(Forward MM, FMM)
* 逆向最大匹配算法(Backward MM, BMM)
* 双向最大匹配算法(Bi-directional MM)
* 问题：给定一个句子 S = c<sub>1</sub>c<sub>2</sub>...c<sub>n</sub>. 假设词：w<sub>i</sub>=c<sub>1</sub>c<sub>2</sub>...c<sub>m</sub>。m为词典中最长词的字数。

## FMM算法描述
* (0) 令i=0，当前指针p<sub>i</sub>指向输入字串的初始位置，执行下面的操作
* (1) 计算当前指针p<sub>i</sub>到字符串末端的字数n，if n=1,转(3)。否则，令m=词典中最长单词的字数，ifn<m，令m=n
* (2) 从当前p<sub>i</sub>起取m个汉字作为词w<sub>i</sub>。做如下判断：
	*  a) 如果w<sub>i</sub>确实是字典中的词。则在w<sub>i</sub>后添加一个切分标志，转c)。
	*  b) 如果w<sub>i</sub>不是词典中的词且w<sub>i</sub>的长度大于1，将w<sub>i</sub>的右端去掉一个字，转(2)中的a)步骤。如果w<sub>i</sub>的长度等于1，则在w<sub>i</sub>后添加一个切分标志，将w<sub>i</sub>作为单字词添加到词典中，执行c)
	*  c) 根据w<sub>i</sub>的长度修改指针p<sub>i</sub>的位置,如果p<sub>i</sub>指向字符串的末端，转(3)，否则，i=i+|w<sub>i</sub>|，返回(1)
* (3) 输出切分结果。

* 优点：
	* 程序简单易行，开发周期短
	* 仅需要很少的语言资源，不需要任何词法、句法、语义资源
* 缺点：
	* 切分歧义消解的能力差
	* 切分正确率不高，一般在95%左右。

