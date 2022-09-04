---
title: "🧑‍💻 Add interactive code into your blog post"
date: 2022-09-05T00:17:00+04:00
draft: false
---

## What is it?
Today I found interesting service named [Replit](https://replit.com) which is kind of **online IDE** for many languages, i already saw many of similar project, for example [ideone](http://ideone.com), which i used in my school days for solving contest on shool computers, where compilers is not installed. 

But **Replit** provide much more, you have a really good editor, **collaborative mode**, you may build complicated projects which contains **multiple files**, using ppular build systems, like **CMake**. 


## Interactive code in blogpost
The main feature, as i think is - your **program may be interactive**, you may embed this code into your blog. For example, I made a simple **Snake game** which you can run right now 👇

`Hint: control WASD on english keyboard only`
{{<replit src="https://replit.com/@alxmamaev/SnakeGameCpp">}}


You may check source code of this game just clic to the button `show files`.

## How to put this demo on your blog

To add repl on your site, just create a own repl on the site, and then add this `iframe` on your web page:

```html
<iframe frameborder="0" width="100%" height="500px"
        src="REPL_URL?embed=true"></iframe>
```

If you have blog on the **Medium**, it's actually not possible, because they are not support user-provided `iframes`, so i recommend you switch to `Hugo`.

Thanks **Erik Smith** which create a [tutorial how to add Replit to hugo](https://developer451.com/post/hugo-replit-shortcode/). 