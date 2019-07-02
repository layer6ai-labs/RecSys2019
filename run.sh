#!/usr/bin/env bash

# Set your Trivago dataset path here (make sure to include trailing "/")
dataPath="/data/recsys2019/data/trivagoRecSysChallengeData2019_v2/"

# Set your output path here (make sure to include trailing "/")
outPath="/data/recsys2019/output/"

# Set your model version here (choose "1" for fast model, choose "2" for accurate model)
modelVersion="1"

# Do not settings touch below this
export MAVEN_OPTS="-Xms200g -Xmx200g"
mvn clean compile
mvn exec:java -Dexec.mainClass="recsys2019.RecSys19DataParser" -Dexec.args="${dataPath} ${outPath}" 2>&1 | tee "${outPath}log_DataParser.txt"
mvn exec:java -Dexec.mainClass="recsys2019.RecSys19Model" -Dexec.args="${dataPath} ${outPath} ${modelVersion} extract" 2>&1 | tee "${outPath}log_ModelExtract.txt"
mvn exec:java -Dexec.mainClass="recsys2019.RecSys19Model" -Dexec.args="${dataPath} ${outPath} ${modelVersion} train" 2>&1 | tee "${outPath}log_ModelTrain.txt"
mvn exec:java -Dexec.mainClass="recsys2019.RecSys19Model" -Dexec.args="${dataPath} ${outPath} ${modelVersion} validate" 2>&1 | tee "${outPath}log_ModelValidate.txt"
mvn exec:java -Dexec.mainClass="recsys2019.RecSys19Model" -Dexec.args="${dataPath} ${outPath} ${modelVersion} submit" 2>&1 | tee "${outPath}log_ModelSubmit.txt"