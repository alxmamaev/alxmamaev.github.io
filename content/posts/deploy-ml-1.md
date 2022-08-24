---
title: "Deploy ML model: Part 1 - Running triton server"
date: 2022-08-24T23:46:30+04:00
draft: true
---

In this series i will tell you how to deploy your ML model to the real production. 

* All code used in this article availiable on [GitHub](https://github.com/alxmamaev/deploy_ml_course).
* If you have any suggestions to improove this article, you can text me hi@alxmamaev.me 

# Why this article exist?

If you found this article, you probably saw a lot of tutorials _"How to deploy your ml model in 5 minutes"_, mostly this tutorial uses Flask as API backend which just call PyTorch model, so actually that is not a best way, it may bee good anought for quick-and-dirty developement, for example hackathons or MVP, but for a real production **it is really bad**, because this solution does not support requests batching, multiple-instance inference, scaling and so on. 

In this series i will tell you a **better way to deploy models**, which i got from my expirience in machine learning. 



# Syllabus
* **Run Triton server** (This article)
* [Data preprocessing pipeline](https://example.com) 
* [Request batching](https://example.com)
* [Performance testing](https://example.com)
* [Docker Swarm](https://example.com)
* [Running model on multiple nodes](https://example.com)
* [C++ preprocessing module in Triton](https://example.com)
* [TensorRT](https://example.com)
* [Model quantization](https://example.com)

