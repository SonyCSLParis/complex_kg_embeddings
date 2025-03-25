for syntax in "simple_rdf_reification" "simple_rdf_sp" "simple_rdf_prop"; do
    for prop in 0 1; do
        for subevent in 0 1; do
            if [ $prop -eq 1 ] || [ $subevent -eq 1 ]; then
                name="kg_base"
                if [ $prop -eq 1 ]; then
                    name="${name}_prop"
                fi
                if [ $subevent -eq 1 ]; then
                    name="${name}_subevent"
                fi
                name="${name}_role_${syntax}"
                if [ ! -d "./data/${name}" ]; then
                    python prep_data.py ./data/${name} --prop ${prop} --subevent ${subevent} --role 1 --role_syntax ${syntax}
                else
                    echo "Directory ./data/${name} already exists, skipping."
                fi
            fi
        done
    done
done
