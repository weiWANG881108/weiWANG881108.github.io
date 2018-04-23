---
layout: post
title: 读书笔记一 术语(Terminology)
description: Effective C++,由侯捷译。
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
    *  类(class)声明式： class Widget;
    *  模版(template)声明式： template\<typename T\> class GraphNode;
{: .notification .is-primary}
*  每个函数的声明揭示其签名式(Signature)，也就是参数和返回类型

## 定义式(Definition)
*  定义式的任务是提供编译器一些声明所遗漏的细节。
    *  对象(Object)定义式，编译器为此对象拨发内存的地点。int x
    *  函数(function)定义式, 提供了代码本体
    *  类(class)定义式，提供了它们的成员
    *  模版(class)定义式，提供了它们的成员
*  初始化(Initialization)： 给予对象初值的过程。
    *  用户自定义类型的对象，初始化由构造函数执行。
        *  default构造函数是一个可被调用而不带任何实参者。
        *  构造函数声明为explicit。可以阻止构造函数被用来执行隐士类型转换(implict type conversion)。
        *  `除非有一个好理由允许构造函数被用于隐式类型转换，否则把它声明为explicit`。

        ```C++
            class B{
                  public:
                  explict B(int x=0, bool b=true);
            };
      
        ```
   *  copy 构造函数：以同类对象初始化自我对象。它定义一个对象如何`passed by value`.
   *  copy assignment操作符：从同类对象中拷贝其值到自我对象。

        ```C++
            class Widget{
                  Widget();
                  Widget(const Widget & rhs); //copy 构造函数
                  Widget& operator=(const Widget & rhs)  //copy assignment操作符
            };
            Widget w1;
            Widget w1(w2);   //copy 构造函数
            Widget w1 = w2;  //copy 构造函数
            w1 = w2          //copy assignment操作符
            // copy构造和copy赋值的区别：如果一个新对象被定义，一定会有个构造函数被调用，不可能调用赋值操作。如果没有新对象被定义，就不会有构造函数被调用。那么当然是赋值操作被调用。
        ```

## STL标准模版库(Standard Template Library)

是C++标准程序库的一部分，致力于容器、迭代器、算法及及相关机能。许多相关机能以函数对象(function objects)实现。那是行为像函数的对象。这样的对象来自于重载operator()的classes。

## 不明确行为(undefined behavior)

    ```C++
            //Example 1
            int *p = 0; // p is a null pointer
            std::cout << *p;
            // Example 2
            char name[] = "Darla";
            char c = name[10]; //无效的数组索引
    ```