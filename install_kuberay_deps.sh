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

# Install both CRDs and KubeRay operator v1.0.0-rc.0.
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0-rc.0

# Apply Clusters-IP
kubectl expose pod $(kubectl get pod -l ray.io/node-type=head | awk 'NR>1 {print $1}') --type=ClusterIP --name=kuberay-metrics-clusterip --port=8080

kubectl expose service prometheus-prometheus-kube-prometheus-prometheus-0 -n prometheus-system --type=ClusterIP --name=prometheus-clusterip --port=9090

kubectl expose deployment prometheus-grafana -n prometheus-system --type=ClusterIP --name=grafana-clusterip --port=3000

kubectl expose service $(kubectl get pod -l ray.io/node-type=head | awk 'NR>1 {print $1}') --type=ClusterIP --name=ray-dashboard-clusterip --port=8265

echo "Complete installation"