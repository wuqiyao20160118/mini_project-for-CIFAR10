# mini_project-for-CIFAR10
Using self-written ResNet18 to classify CIFAR10. It includes Tensorflow and Keras versions.  

    Before using these codes, please create a directory "./datasets".   
    If you are using MACOSX or Linux, please use following codes to download datasets under "./datasets" directory.  
<pre><code>wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz  
tar -xzvf cifar-10-binary.tar.gz  
rm cifar-10-binary.tar.gz
</code></pre>
    If you are a Windows user, you can directly download them on the website.  
    In Tensorflow, there are lots of tricks to be considered. Therefore the code I provide may use some codes in Keras source code especially in data augmentation part. But those parts just contain some math and are easy to understand.  
    It is important to mention that the test accuracy in TensorFlow version is a little bit lower than Keras version, so there may be some little mistakes. I'm very glad if someone can point them out.  
    I also provide VGG16 network as a baseline for ResNet.  
    
# References
You can check the Paper "Deep Residual Learning for Image Recognition" (https://arxiv.org/abs/1512.03385) for more details.  
Some codes draw on the experience of others(like the assignment of CS231n). You can check those on https://github.com/cs231n/cs231n.github.io.
