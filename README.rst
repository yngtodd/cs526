.. raw:: html

    <embed>
        <p align="center">
            <img width="300" src="https://github.com/yngtodd/cs526/blob/master/img/spark_kirby.png">
        </p>
    </embed>

---------------------

===================== 
Spark Notes for CS526
=====================

A bit of information for navigating Spark. The code examples in both the notebooks and modules
are all Python, as I am more familiar with that language compared with Scala. While Scala seems
great, I would personally recommend Python as it is the language de facto for most of machine 
learning. And it is just plain fun. 

Getting Started
---------------

I would highly recommend starting off with a Python distribution from Anaconda_. It is great out of the box and helps manage Python dependencies. 

Quick note: I would recommend getting the most recent version of Python offered. However, I 
have heard that there are some potential issues running Python 3.6 with Pyspark (I will try to
find a reference for that). Just to be on the safe side, I would create a conda virtual 
environment using Python 3.5::

    conda create -n pyspark python=3.5

The above line will create a conda environment named pyspark using Python 3.5. You can name
the environment however you like. Once you have that, stepping into that environment is as
simple as::

    source activate pyspark



.. _Anaconda: https://www.anaconda.com/download/#linux

