#!/bin/bash

# Function to display a message and read user input
function read_choice {
    local message=$1
    local choice
    read -p "$message (yes/no): " choice
    echo $choice
}
choice_scrape=$(read_choice "Do you want to run the scraping script? (Type 'yes' ONLY if you do not have data_scraped.csv yet!)")
choice_sentiment=$(read_choice "Do you want to run the sentiment analysis over all articles all over again? (Type 'yes' ONLY if you do not have data_nlp.csv yet!)")
choice_clickbait=$(read_choice "Do you want to run the clickbait classification script? (You need the scarped data as data_scapred.csv!)")
choice_trends=$(read_choice "Do you want to run the google trends classification over all articles all over again?")

# echo "Updating the requirements..."
# pip install -r requirements_dev.txt

if [ "$choice_scrape" == "yes" ]; then
    echo "+++++ Running data scraping script +++++"
    python scripts/2a_get_df_scraped.py
else
    echo "-----> Skipping data scraping"
fi

if [ "$choice_clickbait" == "yes" ]; then
    echo "+++++ Running data clickbait script +++++"
    python scripts/2c_clickbait_classification.py
else
    echo "-----> Skipping clickbait script"
fi

echo ""
echo "+++++ Combining data deliveries +++++"
python scripts/1_merge_source.py

echo ""
echo "+++++ Aggregating by page_id and date +++++"
python scripts/2b_get_df_aggr.py

echo ""
echo "+++++ Aggregating by page_id +++++"
python scripts/3_page_id_agg.py

if [ "$choice_trends" == "yes" ]; then
    echo "+++++ Running google trends classification script +++++"
    python scripts/3B_trends_classification.py
else
    echo "-----> Skipping trends classification"
fi

echo ""
echo "+++++ Extracting features +++++"
python scripts/4_get_df_features.py

if [ "$choice_sentiment" == "yes" ]; then
    echo "+++++ Running sentiment analysis script +++++"
    python scripts/5_sentiment_analysis.py

    echo ""
    echo "+++++ Running clickbait classification script +++++"
    python scripts/preprocessing_NLP.py
    python scripts/clickbait_classification.py
else
    echo "-----> Skipping sentiment analysis"
    python scripts/5A_sentiment_merge.py

fi
echo ""
echo "+++++ Prettifying the data segments for the D-Drivers Data App +++++"
python scripts/6_prepare_for_demo.py