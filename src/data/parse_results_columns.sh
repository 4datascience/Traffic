if [ $# -ne 1 ]
then
    echo "Must pass as first argument the file to parse";
    exit
fi

echo "Parsing  $1";
sed -i '/LINK/d' $1
sed -i '/Implementation/d' $1
sed -i 's/^.*Recall //' $1
sed -i 's/\%//' $1
sed -i 's/\./\,/' $1
cat $1
