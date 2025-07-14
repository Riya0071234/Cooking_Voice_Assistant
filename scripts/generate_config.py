# scripts/generate_config.py
"""
A utility script to programmatically generate a perfect config.yaml file.

This script eliminates any manual formatting or indentation errors by
defining the entire configuration structure in a Python dictionary and
dumping it to a YAML file with correct syntax.

Run this script ONCE to create the definitive configuration.
"""
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s')

# This dictionary defines the entire, correct structure of your config file.
# Note that 'youtube' is correctly nested inside 'contextual_sources'.
config_data = {
    "database": {
        "url": "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${RDS_HOSTNAME}:${DB_PORT}/${POSTGRES_DB}"
    },
    "api_keys": {
        "openai": "${OPENAI_API_KEY}",
        "youtube": "${YOUTUBE_API_KEY}",
        "reddit_client_id": "${REDDIT_CLIENT_ID}",
        "reddit_client_secret": "${REDDIT_CLIENT_SECRET}",
        "reddit_user_agent": "CookingAssistantScraper/1.0 by YourUsername"
    },
    "rag": {
        "embedding_model": "text-embedding-3-small",
        "completion_model": "gpt-4-turbo"
    },
    "recipe_sites": {
        "indian": ["https://www.archanaskitchen.com", "https://www.tarladalal.com", "https://www.vegrecipesofindia.com",
                   "https://hebbarskitchen.com", "https://www.manjulaskitchen.com", "https://www.sanjeevkapoor.com",
                   "https://www.indianhealthyrecipes.com", "https://ministryofcurry.com"],
        "pan_asian": ["https://www.chinasichuanfood.com", "https://www.justonecookbook.com",
                      "https://hot-thai-kitchen.com", "https://rasamalaysia.com", "https://www.maangchi.com",
                      "https://thewoksoflife.com", "https://www.vietworldkitchen.com"],
        "italian_mexican": ["https://www.giallozafferano.it", "https://www.mexicanplease.com",
                            "https://cookieandkate.com", "https://www.isabeleats.com", "https://www.skinnytaste.com",
                            "https://sipandfeast.com"],
        "western_international": ["https://www.allrecipes.com", "https://www.seriouseats.com",
                                  "https://www.bonappetit.com", "https://www.kingarthurbaking.com/recipes",
                                  "https://smittenkitchen.com", "https://www.foodandwine.com",
                                  "https://sallysbakingaddiction.com", "https://www.onceuponachef.com"]
    },
    "contextual_sources": {
        "social_media": {
            "instagram": {"enabled": True, "scrape_comments": True,
                          "accounts": ["nishamadhulika", "kabitaskitchen", "ranveer.brar", "hebbars.kitchen",
                                       "bonappetitmag", "seriouseats", "sanjeevkapoor", "food52", "smittenkitchen",
                                       "maangchi"],
                          "hashtags": ["cookingfail", "kitchennightmare", "cookinghelp", "recipequestion", "burntfood",
                                       "bakingfail", "askchef", "cookingadvice"]},
            "facebook": {"enabled": False, "scrape_comments": True,
                         "pages": ["RanveerBrarOfficial", "HebbarsKitchen", "SeriousEats", "NYTCooking"],
                         "groups": ["foodbloggersofindia", "chefathome"]}
        },
        "youtube": {
            "scrape_comments": True, "max_results_per_channel": 30,
            "channels": {
                "indian": ["UCBFKdfGk_f_qf9S22i5Qp2g", "UCe1mJcsRr9Jal9vpaQfS14g", "UCPLeqiN-FFK_M-Q-sX-H-4g",
                           "UCDi81b9G0j0iDDp3b0iGBpQ", "UC4w2b9_n0C-P88VR3GAl4pQ", "UC3P5Afsrkhy2lT_c0s1G2gQ"],
                "pan_asian": ["UCCYc7s_pXp_d22a7g6Kq2aA", "UCp2_pQ0sE5so2uI92i3G1bg", "UC54SL_45d7-4w0H2m2_hb8g",
                              "UC8EMp8n_la4f-pEaREy9qNQ"],
                "italian_mexican": ["UCW_wx72fO_2i0gAq_3V3Lqg", "UCL1w6-zY2a-j-vH-5V_s-tQ", "UCJmG_8iZ4_u-bT-K9E72S1A",
                                    "UCTO2cT1HwNWBgE_3D4yvJgQ"],
                "western": ["UCzH3iADRIq1IJlIXjfNgTpA", "UCAb2lksTv2r2_2a_2I426JA", "UCkY5ORCN22s3fbbqH32_Isw",
                            "UCpprB-U_3_A-holD_yD2GqA", "UCT3v2w_3yG_g4g2s-2G_33w"]
            }
        },
        "forums": {
            "reddit": {"enabled": True, "scrape_comments": True,
                       "subreddits": ["Cooking", "AskCulinary", "IndianFood", "recipes", "Baking",
                                      "cookingforbeginners", "chefit", "KitchenConfidential", "whatscooking"]},
            "quora": {"enabled": True, "scrape_answers": True,
                      "topics": ["Cooking", "Indian-Cooking", "Recipes", "Food", "Baking", "Spices", "Kitchen-Hacks",
                                 "Food-Preservation", "Culinary-Arts"]}
        }
    },
    "scraping": {
        "delay_between_requests": 2, "max_retries": 3, "timeout": 30, "concurrent_workers": 4,
        "contextual_keywords": ["help", "problem", "mistake", "fix", "burnt", "undercooked", "overcooked", "salty",
                                "bland", "curdled", "soggy", "substitute", "alternative", "why did", "how to",
                                "what went wrong"]
    },
    "processing": {
        "deduplication_similarity_threshold": 0.9,
        "auto_tagging": {"enabled": True, "strategy": "dynamic_tfidf_clustering",
                         "params": {"max_tags_per_item": 10, "min_word_length": 3, "top_n_keywords_per_cluster": 5}}
    },
    "vision_data": {
        "enabled": True, "yolo_model_path": "models/vision/yolov8_ingredients.pt",
        "confidence_threshold": 0.6, "frame_sampling_interval": 3
    },
    "images": {
        "download_enabled": True, "max_size_bytes": 5242880, "formats": ["jpg", "jpeg", "png"]
    },
    "storage": {
        "raw_data_path": "data/raw", "processed_data_path": "data/processed",
        "contextual_data_path": "data/contextual", "vision_data_path": "data/vision",
        "images_path": "data/images", "log_path": "logs/pipeline.log"
    },
    "validation": {
        "recipe_entry": {"title": {"min_length": 5}, "ingredients": {"min_count": 3, "max_count": 50},
                         "instructions": {"min_count": 3, "max_count": 50}},
        "contextual_entry": {"question": {"min_length": 15, "max_length": 500},
                             "answer": {"min_length": 20, "max_length": 5000}, "tags": {"min_count": 1},
                             "language": {"accepted": ["en", "hi", "hi-en"]}}
    },
    "training": {
        "enabled": True, "openai_base_model": "gpt-3.5-turbo",
        "fine_tuned_model_id": "ft:gpt-3.5-turbo-xxxxxxxx",
        "dataset_path": "data/processed/lang_tagged_scraped_contextual_posts.json"
    },
    "vision_training": {
        "enabled": True, "labeled_dataset_path": "data/vision/labeled_dataset",
        "output_model_path": "models/vision/cooking_stage_classifier.pth",
        "base_model": "efficientnet_b0", "learning_rate": 0.001, "num_epochs": 10, "batch_size": 32
    }
}


def create_config_file():
    """Generates the config.yaml file from the dictionary above."""
    # The config file should be in `config/config.yaml` relative to the project root.
    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)
    config_file_path = config_dir / "config.yaml"

    try:
        with open(config_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, sort_keys=False, indent=2)
        logging.info(f"✅ Successfully created a perfect config file at: {config_file_path}")
    except Exception as e:
        logging.error(f"Failed to create config file: {e}")


if __name__ == "__main__":
    create_config_file()