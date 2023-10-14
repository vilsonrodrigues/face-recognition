#!/bin/bash

# install helm chart
helm repo add qdrant https://qdrant.github.io/qdrant-helm
helm repo update
helm upgrade -i qdrant qdrant/qdrant

echo "Complete Qdrant installation"