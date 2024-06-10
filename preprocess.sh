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
    echo "+++++ Running data scraping script 01_get_df_scraped+++++"
    python scripts/01_get_df_scraped.py
    echo ""
else
    echo "-----> Skipping data scraping"
    echo ""
fi

if [ "$choice_clickbait" == "yes" ]; then
    echo "+++++ Running data clickbait script 02_clickbait_classification+++++"
    python scripts/02_clickbait_classification.py
    echo ""
else
    echo "-----> Skipping clickbait script"
fi

echo ""
echo "+++++ Combining data deliveries 11_merge_source +++++"
python scripts/11_merge_source.py

echo ""
echo "+++++ Aggregating by page_id and date 12_get_df_aggr +++++"
python scripts/12_get_df_aggr.py

echo ""
echo "+++++ Aggregating by page_id 13_page_id_agg +++++"
python scripts/13_page_id_agg.py

echo ""
echo "+++++ Extracting features 20_get_df_features +++++"
python scripts/20_get_df_features.py

if [ "$choice_trends" == "yes" ]; then
    echo "+++++ Running google trends classification script 30_trends_classification (this can take many hours if not executed with hardware accelleration like google colab T4+++++"
    python scripts/30_trends_classification.py
    echo ""
else
    echo "-----> Skipping trends classification"
fi

echo ""
echo "+++++ Adding google trends to features 31_trends_merge +++++"
python scripts/31_trends_merge.py

if [ "$choice_sentiment" == "yes" ]; then
    echo "+++++ Running sentiment analysis script 40_sentiment_analysis +++++"
    python scripts/40_sentiment_analysis.py
    echo ""
else
    echo "-----> Skipping sentiment analysis"
fi

echo ""
echo "+++++ Adding sentiment to features 41_sentiment_merge +++++"
python scripts/41_sentiment_merge.py
echo ""

echo ""
echo "+++++ Prettifying the data segments for the D-Drivers Data App 50_prepare_for_demo +++++"
python scripts/50_prepare_for_demo.py
echo ""