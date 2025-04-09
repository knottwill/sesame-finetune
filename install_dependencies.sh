#!/bin/bash

# CSM repository details
CSM_REPO="https://github.com/SesameAILabs/csm.git"
CSM_COMMIT="836f886515f0dec02c22ed2316cc78904bdc0f36"
TEMP_DIR=$(mktemp -d)

# Clone and checkout CSM
git clone $CSM_REPO $TEMP_DIR
cd $TEMP_DIR
git checkout $CSM_COMMIT

# Install CSM and dependencies
pip install -e .
pip install wandb==0.19.6 pandas tqdm

# Cleanup
cd -
rm -rf $TEMP_DIR 