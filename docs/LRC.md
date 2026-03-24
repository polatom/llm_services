Welcome to ÚFAL LRC (Linguistic Research Cluster)

The cluster is built on top of SLURM and is using Lustre for data storage. 

Access to LRC

To access the cluster you need an account in the UFAL network. Those not affiliated with the department as a student or employee need an approval from the head of UFAL for creating the account. 

If you have an account you can use SSH from inside of the UFAL network. Every machine in the domain ufal.hide.ms.mff.cuni.cz can be accessed directly from the UFAL network. When working from home/Eduroam you need to connect to the UFAL network first.
Submit nodes

To submit a job you need to log in to one of the following submit nodes:

  lrc[12].ufal.hide.ms.mff.cuni.cz
  sol[1-8].ufal.hide.ms.mff.cuni.cz

Here we are using compact notation, in which square brackets (i.e. [ and ]) are used to delimit ranges of numeric values. Therefore, there are 2 lrc nodes and 8 sol nodes. With a bit of simplification, a node means a machine (i.e. a computer) in a cluster.

It is not allowed to run computations on the lrc nodes.
Basic usage

The following tutorial is meant to provide only a simplified overview of the cluster usage. Reading further documentation is strongly recommended before running any serious experiments. You should always at least try to guess how many resources your job will consume and set the requirements accordingly. To avoid unexpected failures please make sure your disk quota is not exceeded.

The core idea is to write a batch script containing the commands you wish to run and a list of SBATCH directives specifying the resources or parameters that you need for your job. Then the script is submitted to the cluster with the command:

  sbatch myJobScript.sh

Here is a simple working example:

  #!/bin/bash
  #SBATCH -J helloWorld      # name of job
  #SBATCH -p cpu-troja       # name of partition or queue (default=cpu-troja)
  #SBATCH -o helloWorld.out  # name of output file for this submission script
  #SBATCH -e helloWorld.err  # name of error file for this submission script
  
  # run my job (some executable)
  sleep 5
  echo "Hello I am running on cluster!"

After submitting this simple code you should end up with the two files (helloWorld.out and helloWorld.err) in the directory where you called the sbatch command. 