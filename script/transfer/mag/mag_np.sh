#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --cluster=htc
#SBATCH --partition=scavenger
#SBATCH --account=hdaqing

#SBATCH --job-name=mag_np
#SBATCH --output=slurm_output/mag_np.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=1-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

source ~/.bash_profile # reload LD_LIBRARY due to error ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found

input_dir="/zfs1/hdaqing/rum20/kp/data/kp/oag_v1_cs_nokp/"
output_dir="/zfs1/hdaqing/rum20/kp/data/kp/oag_v1_cs_nokp_np/"

filenames=(train_101.json train_102.json train_103.json train_104.json train_105.json train_106.json train_107.json train_108.json train_109.json train_110.json train_111.json train_112.json train_113.json train_114.json train_115.json train_116.json train_117.json train_118.json train_119.json train_120.json)
# train_0.json train_1.json train_2.json train_3.json train_4.json train_5.json train_6.json train_7.json train_8.json train_9.json train_10.json train_11.json train_12.json train_13.json train_14.json train_15.json train_16.json train_17.json train_18.json train_19.json train_20.json train_21.json train_22.json train_23.json train_24.json train_25.json train_26.json train_27.json train_28.json train_29.json train_30.json train_31.json train_32.json train_33.json train_34.json train_35.json train_36.json train_37.json train_38.json train_39.json train_40.json train_41.json train_42.json train_43.json train_44.json train_45.json train_46.json train_47.json train_48.json train_49.json train_50.json train_51.json train_52.json train_53.json train_54.json train_55.json train_56.json train_57.json train_58.json train_59.json train_60.json train_61.json train_62.json train_63.json train_64.json train_65.json train_66.json train_67.json train_68.json train_69.json train_70.json train_71.json train_72.json train_73.json train_74.json train_75.json train_76.json train_77.json train_78.json train_79.json train_80.json train_81.json train_82.json train_83.json train_84.json train_85.json train_86.json train_87.json train_88.json train_89.json train_90.json train_91.json train_92.json train_93.json train_94.json train_95.json train_96.json train_97.json train_98.json train_99.json train_100.json train_101.json train_102.json train_103.json train_104.json train_105.json train_106.json train_107.json train_108.json train_109.json train_110.json train_111.json train_112.json train_113.json train_114.json train_115.json train_116.json train_117.json train_118.json train_119.json train_120.json

CURDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TEMPLATE_PATH="$CURDIR/mag_np_sbatch.sh"

for filename in "${filenames[@]}"
do
    echo $filename

    EXP_NAME="mag-np-$filename"
    DUMP_SCRIPT_PATH="$CURDIR/tmp/$EXP_NAME.sh"
    rm -f $DUMP_SCRIPT_PATH

    input_path="/zfs1/hdaqing/rum20/kp/data/kp/oag_v1_cs_nokp/$filename"
    output_path="/zfs1/hdaqing/rum20/kp/data/kp/oag_v1_cs_nokp_np/"
    replaces="s/{job_name}/$EXP_NAME/;";
    replaces="$replaces s|{input_path}|$input_path|g;";
    replaces="$replaces s|{output_dir}|$output_dir|g;";
    cat ${TEMPLATE_PATH} | sed -e "$replaces" > ${DUMP_SCRIPT_PATH}

    sbatch $DUMP_SCRIPT_PATH

done
