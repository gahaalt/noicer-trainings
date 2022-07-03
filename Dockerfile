FROM tensorflow/tensorflow:2.9.0-gpu

RUN pip install ipython tensorflow-datasets pydot
RUN apt-get install graphviz -y
