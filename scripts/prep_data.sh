config_count=0
for prop in 0 1; do
    for subevent in 0 1; do
        for role in 0 1; do
            for causation in 0 1; do
                if [ "$causation" -eq 1 ] || [ "$role" -eq 1 ]; then
                    syntax_list="simple_rdf_reification simple_rdf_sp simple_rdf_prop"
                else
                    syntax_list="simple_rdf_prop"
                fi
                
                for syntax in ${syntax_list}; do
                    name="kg_base_prop_${prop}_subevent_${subevent}_role_${role}_causation_${causation}_syntax_${syntax}"
                    printf "Configuration %2d: %s\n" ${config_count} "${name}"
                    config_count=$((config_count+1))
                    if [ ! -d "./data/${name}" ]; then
                        python prep_data.py ./data/${name} --prop ${prop} --subevent ${subevent} --role ${role} --causation ${causation} --syntax ${syntax} 
                    else
                        echo "Directory ./data/${name} already exists, skipping."
                    fi

                    echo "----------------------------------------"
                done
            done
        done
    done
done
printf "Total configurations: ${config_count}"
printf "Done."
