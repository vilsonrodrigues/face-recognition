#!/bin/bash

# Install Prometheus and Grafana #

# Install Prometheus and Grafana
git clone https://github.com/ray-project/kuberay.git
./kuberay/install/prometheus/install.sh

# Remove kuberay folder
rm -r ./kuberay