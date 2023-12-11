#!/bin/bash

# Install Kuberay, Prometheus and Grafana #

# Install Prometheus and Grafana
git clone https://github.com/ray-project/kuberay.git
./kuberay/install/prometheus/install.sh

# Remove kuberay folder
rm -r ./kuberay

# add Kuberay Helm
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Install both CRDs and KubeRay operator v1.0.0
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0

echo "Complete installation"