printf 'are you sure to delete the logs (y/n)? '
read answer

if [ "$answer" != "${answer#[Yy]}" ] ;then 
    rm -r logs/*
    mkdir logs/z_archive
else
    echo No
fi

