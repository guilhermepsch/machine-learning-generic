import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataVisualizationConfig:
    plot_output_path: str = os.path.join('artifacts', 'tsne_plot.png')
    distribution_output_path: str = os.path.join('artifacts', 'images_per_syndrome_distribution.png')

class DataVisualization:
    def __init__(self):
        self.visualization_config = DataVisualizationConfig()

    def visualize_data(self, data: pd.DataFrame, feature_columns: list, hue_column: str = 'syndrome_id'):
        try:
            logging.info("Starting data visualization process")
            df_tsne = self.apply_tsne(data, feature_columns)
            df_tsne[hue_column] = data[hue_column].values
            self.plot_tsne(df_tsne, hue_column)
            self.plot_images_per_syndrome_distribution(df, syndrome_column='syndrome_id')
            logging.info("Data visualization completed successfully")
        except Exception as e:
            raise CustomException(f"Error in data visualization: {str(e)}", sys)
          
    def apply_tsne(self, data: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        try:
            logging.info("Standardizing the data before applying t-SNE")
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(data[feature_columns])
            logging.info("Applying t-SNE to reduce dimensionality to 2D")
            tsne = TSNE(perplexity=45, n_components=2, random_state=42, n_iter=100000)
            embeddings_2d_tsne = tsne.fit_transform(embeddings_scaled)
            df_tsne = pd.DataFrame(embeddings_2d_tsne, columns=['tSNE1', 'tSNE2'])
            
            return df_tsne
        except Exception as e:
            raise CustomException(f"Error in applying t-SNE: {str(e)}", sys)

    def plot_tsne(self, df_tsne: pd.DataFrame, hue_column: str = 'syndrome_id'):
        try:
          # Colors
          unique_syndrome_ids = df_tsne['syndrome_id'].unique()
          palette = sns.color_palette("tab20", len(unique_syndrome_ids))
          id_to_color = {id_: palette[i] for i, id_ in enumerate(unique_syndrome_ids)}
          df_tsne['color'] = df_tsne['syndrome_id'].map(id_to_color)
          
          logging.info("Plotting the t-SNE results")
          plt.figure(figsize=(10, 6))
          sns.scatterplot(data=df_tsne, x='tSNE1', y='tSNE2', hue=hue_column, palette=id_to_color, s=100)
          plt.title('t-SNE of Embeddings Colored by Syndrome ID')
          plt.xlabel('t-SNE Component 1')
          plt.ylabel('t-SNE Component 2')
          plt.legend(title=hue_column)
          
          plt.savefig(self.visualization_config.plot_output_path)
          plt.close()
          logging.info(f"Plot saved at {self.visualization_config.plot_output_path}")
        except Exception as e:
          raise CustomException(f"Error in plotting t-SNE: {str(e)}", sys)

    def provide_dataset_statistics(self, data: pd.DataFrame, syndrome_column: str = 'syndrome_id'):
        try:
            logging.info("Generating basic dataset statistics")
            
            unique_syndromes = data[syndrome_column].nunique()
            logging.info(f"Number of unique syndromes: {unique_syndromes}")
            
            images_per_syndrome = data[syndrome_column].value_counts()
            logging.info(f"Number of images per syndrome:")
            logging.info(images_per_syndrome)
            
            return images_per_syndrome
        
        except Exception as e:
            raise CustomException(f"Error in providing dataset statistics: {str(e)}", sys)

    def plot_images_per_syndrome_distribution(self, data: pd.DataFrame, syndrome_column: str = 'syndrome_id'):
        try:
            images_per_syndrome = self.provide_dataset_statistics(data, syndrome_column)
            
            logging.info("Plotting the distribution of images per syndrome")
            
            plt.figure(figsize=(20, 12))
            sns.barplot(x=images_per_syndrome.index, y=images_per_syndrome.values, palette="viridis")
            plt.title('Distribution of Images per Syndrome')
            plt.xlabel('Syndrome ID')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45)
            
            plt.savefig(self.visualization_config.distribution_output_path)
            plt.close()
            logging.info(f"Images per syndrome distribution plot saved at {self.visualization_config.distribution_output_path}")
        except Exception as e:
            raise CustomException(f"Error in plotting the distribution of images per syndrome: {str(e)}", sys)

if __name__ == "__main__":
    try:
        df = pd.read_csv('artifacts/data.csv')
        obj = DataVisualization()
        feature_columns = [f'dim_{i}' for i in range(320)]
        obj.visualize_data(df, feature_columns)
    
    except Exception as e:
        logging.error(f"Error in the main execution: {str(e)}")
