
from os.path import join
import sys

SLURM_CONFIGURATION = """#!/bin/bash
#SBATCH --job-name=%s
#SBATCH -o %s.log
#SBATCH -c 1
#SBATCH -p short
#SBATCH -t 10:00:00
#SBATCH --mem=120G
srun python ir_undersampling_test.py %s %s %s
"""


FILEMODEL2BESTK = {
("criteo-uplift.csv0.pickle.gz", "cvtrf"): 150,
("criteo-uplift.csv0.pickle.gz", "dcrf"): 46,
("criteo-uplift.csv0.pickle.gz", "dc"): 7,
("criteo-uplift.csv0.pickle.gz", "cvt"): 201,
("criteo-uplift.csv155768161.pickle.gz", "cvtrf"): 230,
("criteo-uplift.csv155768161.pickle.gz", "dcrf"): 6,
("criteo-uplift.csv155768161.pickle.gz", "dc"): 38,
("criteo-uplift.csv155768161.pickle.gz", "cvt"): 181,
("criteo-uplift.csv1740898770.pickle.gz", "cvtrf"): 230,
("criteo-uplift.csv1740898770.pickle.gz", "dcrf"): 21,
("criteo-uplift.csv1740898770.pickle.gz", "dc"): 31,
("criteo-uplift.csv1740898770.pickle.gz", "cvt"): 171,
("criteo-uplift.csv1.pickle.gz", "cvtrf"): 210,
("criteo-uplift.csv1.pickle.gz", "dcrf"): 21,
("criteo-uplift.csv1.pickle.gz", "dc"): 18,
("criteo-uplift.csv1.pickle.gz", "cvt"): 201,
("criteo-uplift.csv2316060921.pickle.gz", "cvtrf"): 230,
("criteo-uplift.csv2316060921.pickle.gz", "dcrf"): 41,
("criteo-uplift.csv2316060921.pickle.gz", "dc"): 33,
("criteo-uplift.csv2316060921.pickle.gz", "cvt"): 231,
("criteo-uplift.csv2495284092.pickle.gz", "cvtrf"): 210,
("criteo-uplift.csv2495284092.pickle.gz", "dcrf"): 11,
("criteo-uplift.csv2495284092.pickle.gz", "dc"): 14,
("criteo-uplift.csv2495284092.pickle.gz", "cvt"): 201,
("criteo-uplift.csv2.pickle.gz", "cvtrf"): 150,
("criteo-uplift.csv2.pickle.gz", "dcrf"): 16,
("criteo-uplift.csv2.pickle.gz", "dc"): 42,
("criteo-uplift.csv2.pickle.gz", "cvt"): 231,
("criteo-uplift.csv3147263977.pickle.gz", "cvtrf"): 210,
("criteo-uplift.csv3147263977.pickle.gz", "dcrf"): 16,
("criteo-uplift.csv3147263977.pickle.gz", "dc"): 3,
("criteo-uplift.csv3147263977.pickle.gz", "cvt"): 191,
("criteo-uplift.csv3.pickle.gz", "cvtrf"): 170,
("criteo-uplift.csv3.pickle.gz", "dcrf"): 16,
("criteo-uplift.csv3.pickle.gz", "dc"): 3,
("criteo-uplift.csv3.pickle.gz", "cvt"): 171,
("criteo-uplift.csv4.pickle.gz", "cvtrf"): 170,
("criteo-uplift.csv4.pickle.gz", "dcrf"): 21,
("criteo-uplift.csv4.pickle.gz", "dc"): 30,
("criteo-uplift.csv4.pickle.gz", "cvt"): 231,
}

if __name__ == '__main__':
    print("Generating SLURM jobs for pickles from a given directory.")

    try: 
        directory = sys.argv[1]
    except:
        print("An argument required - a directory containing (gzipped) pickles!")
        sys.exit(-1)

    jobs = []
    for (filename, model), best_k in FILEMODEL2BESTK.items():
        path = join(directory, filename)

        kstring = "%i" % best_k
        identifier = "%s_%s_%s" % (filename, model, kstring)
        open(identifier+".job", "w").write(
            SLURM_CONFIGURATION % (identifier, identifier, path, model, kstring)
        )
        jobs.append( identifier+".job" )


    open("RUN_ALL_SLURM_JOBS.sh", "w").write("sbatch "+"\nsbatch ".join(jobs)+"\n\n")
    print("Done. To run all jobs execute: sh RUN_ALL_SLURM_JOBS.sh")
