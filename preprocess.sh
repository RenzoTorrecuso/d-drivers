#!/bin/bash

# Function to display a message and read user input
function read_choice {
    local message=$1
    local choice
    read -p "$message (yes/no): " choice
    echo $choice
}
choice_scrape=$(read_choice "Do you want to run the scraping script? (Type 'yes' ONLY if you do not have data_scraped.csv yet!)")
choice_clickbait=$(read_choice "Do you want to run the clickbait classification script? (Type 'yes' ONLY if you do not have clickbait.csv yet! You need the scarped data as data_scapred.csv!)")
choice_trends=$(read_choice "Do you want to run the google trends classification over all articles all over again?")
choice_sentiment=$(read_choice "Do you want to run the sentiment analysis over all articles all over again? (Type 'yes' ONLY if you do not have data_nlp.csv yet!)")

# echo "Updating the requirements..."
# pip install -r requirements_dev.txt

if [ "$choice_scrape" == "yes" ]; then
    echo "+++++ Running data scraping script +++++"
    python scripts/01_get_df_scraped.py
else
    echo "-----> Skipping data scraping"
fi

if [ "$choice_clickbait" == "yes" ]; then
    echo "+++++ Running data clickbait script +++++"
    python scripts/02_clickbait_classification.py
else
    echo "-----> Skipping clickbait script"
fi

echo ""
echo "+++++ Combining data deliveries +++++"
python scripts/11_merge_source.py

echo ""
echo "+++++ Aggregating by page_id and date +++++"
python scripts/12_get_df_aggr.py

echo ""
echo "+++++ Aggregating by page_id +++++"
python scripts/13_page_id_agg.py

echo ""
echo "+++++ Extracting features +++++"
python scripts/20_get_df_features.py

if [ "$choice_trends" == "yes" ]; then
    echo "+++++ Running google trends classification script (this can take many hours if not executed with hardware accelleration like google colab T4+++++"
    python scripts/30_trends_classification.py
else
    echo "-----> Skipping trends classification"
fi

echo ""
echo "+++++ Adding google trends to features +++++"
python scripts/31_trends_merge.py

if [ "$choice_sentiment" == "yes" ]; then
    echo "+++++ Running sentiment analysis script +++++"
    python scripts/40_sentiment_analysis.py
else
    echo "-----> Skipping sentiment analysis"
fi

echo ""
echo "+++++ Adding sentiment to features +++++"
python scripts/41_sentiment_merge.py

echo ""
echo "+++++ Prettifying the data segments for the D-Drivers Data App +++++"
python scripts/50_prepare_for_demo.py