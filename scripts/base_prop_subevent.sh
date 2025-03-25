
# for model in transe complex distmult rgcn
if [ "$1" = "cuda:0" ]; then
    models="distmult"
elif [ "$1" = "cuda:1" ]; then
    models="complex"
else
    echo "Error: \$1 must be either 'cuda:0' or 'cuda:1'" >&2
    exit 1
fi

for model in $models ;do
    for base in kg_base_prop kg_base_subevent kg_base_prop_subevent ; do
        for syntax in "simple_rdf_reification" "simple_rdf_sp" "simple_rdf_prop" ; do
            data="${base}_role_${syntax}"
            log_file="./hpo/$model/${data}/logs.json"
            if [ ! -f "$log_file" ]; then
                printf "DATA: $data\tMODEL: $model\n"
                python hpo.py ./data/"$data" "$model" 50 ./hpo --device "$1"
                printf "=======\n"
            else
                printf "Skipping: DATA: $data\tMODEL: $model (logs.json exists)\n"
            fi
        done
    done
done