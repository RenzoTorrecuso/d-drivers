#!/bin/bash

echo "Updating the requirements..."
pip install -r requirements_dev.txt

read -p "Do you want to run the scraping script? (type yes ONLY if you do not have data_scraped.csv yet!) (yes/n): " choice
if [ "$choice" == "yes" ]; then
    python scripts/2a_get_df_scraped.py
    # Add more scripts here if needed
else
    echo "-----> Skipping data scraping "
fi
echo ""
echo "+++++++++++++++++++++++ Combining data deliveries +++++++++++++++++++++++"
python scripts/1_merge_source.py
echo ""
echo "+++++++++++++++++++++++ Aggregating by page_id and date +++++++++++++++++++++++"
python scripts/2b_get_df_aggr.py
echo ""
echo "+++++++++++++++++++++++ Aggregating by page_id +++++++++++++++++++++++"
python scripts/3_page_id_agg.py
echo ""
echo "+++++++++++++++++++++++ Extracting features +++++++++++++++++++++++"
python scripts/4_get_df_features.py