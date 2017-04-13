# Integration among Ophidia and COMPSs

This repository contains applications integrating Ophidia and COMPSs:

- *K-means* : executes K-means clustering based on Lloyd's algorithm on a random data cube exploiting Ophidia and COMPSs

# Setup

In order to run the application a COMPSs installation and the latest PyOphidia version are required. Additionally, a running Ophidia endpoint is required. The application is compatible with Python version 2 (2.6 or 2.7).

For info about COMPSs and the installation procedure, see the [COMP Superscalar web page](https://www.bsc.es/research-and-development/software-and-apps/software-list/comp-superscalar).

For info about Ophidia and the installation procedure, see the [Ophidia framework web page](http://ophidia.cmcc.it).

# Run applications

To run K-means application, for example, launch the following (with the proper credentials of an Ophidia endpoint):

	$ runcompss --pythonpath=$PWD/  $PWD/k-means.py -u <username> -p <password> -H <hostname> -P <port>

