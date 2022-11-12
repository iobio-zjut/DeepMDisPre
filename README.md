# DeepMDisPre
  Prediction of inter-residue multiple distances for protein multiple conformations.

# Developer

```
 Fujin Zhang, Zhangwei Li, and Kailong Zhao 
 College of Information Engineering
 Zhejiang University of Technology, Hangzhou 310023, China
 Email: 2112003036@zjut.edu.cn, lzw@zjut.edu.cn, zhaokailong@zjut.edu.cn
```

# Contact
```
 Guijun Zhang, Prof
 College of Information Engineering
 Zhejiang University of Technology, Hangzhou 310023, China
 Email: zgj@zjut.edu.cn
```

# Installation
- Python > 3.7
- PyTorch 1.3
- Tested on Ubuntu 20.04 LTS

# Running

```
  DeepMDisPre.sh input.fasta output_path
  
  arguments:
  input.fasta                input fasta file
  output_path                path to the output folder (the predicted distance file is in output_path/dis_npz)
```

# Example

```
  bash DeepMDisPre.sh example/seq.fasta output_path
```

# Resources
- DeepMDisPre generate MSA by searching the UniRef30_2020_03_hhsuite, which can be accessed through https://wwwuser.gwdg.de/~compbiol/uniclust.
