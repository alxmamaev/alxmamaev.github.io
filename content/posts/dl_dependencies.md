---
title: "Why you should stop putting PyTorch into requirements.txt"
date: 2022-08-09T00:17:41+04:00
draft: true
---

Im working alot with different open-source machine-learning project, thats have many different approach to controll dependencies. 

Usually you create `requirements.txt` while where you list all libraries that you need with their versions. 
The idea is simple, you need just run `pip install -r requirements.txt` to install all you need, thats working in most cases, but it's breaking all when you are working on some project which using Pytorch.

Lets me give you few examples. You have an `requirements.txt` which contains `torch==1.8.0` as an dependencies

## Case №1
We have an done developement enviroment with installed latest CUDA 11.6 



TEST
