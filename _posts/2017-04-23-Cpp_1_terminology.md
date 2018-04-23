---
layout: post
title: 读书笔记一 术语(Terminology)
description: Effective C++
categories:
    - Cpp
comments: true
permalink: 
---
Effective C++ 读书笔记一：术语(Terminology)

## 声明式(Declaration)
*  告诉编译器某个东西的名称和类型(type)，但略去细节
    *  对象(Object)声明式： extern int x
    *  函数(function)声明式： std::size_t numDigits(int number)
    *  类声(class)明式： class Widget;
    *  模版(template)声明式： template<\typename T> class GraphNode;
{: .notification .is-primary}
*  每个函数的声明揭示其签名式(Signature)，也就是参数和返回类型
*  
