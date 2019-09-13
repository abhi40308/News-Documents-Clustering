# News-Documents-Clustering
News documents clustering using latent semantic analysis. Used LSA and K-means algorithms to cluster news documents and visualized the results using UMAP (Uniform Manifold Approximation and Projection).   

Considering the frequency(tf-idf) of important words in the news documents, the news documents are clustered where the related documents are shown using the same color which can be seen in the screenshots in the end. The color is decided by using k-means(running k-means on data separately and giving integer values to each documents based on k-means similarity results) and the actual positioning of documents(each document is represented by a dot on the graph) is achieved by applying LSA, thus verifying the results obtained using k-means.


This code is part of [medium blog post](https://medium.com/@abhijeet40308/news-documents-clustering-using-python-latent-semantic-analysis-b95c7b68861c)  
This post was published in [mc.ai](https://mc.ai/news-documents-clustering-using-python-latent-semantic-analysis/)  
Link to [google colab](https://colab.research.google.com/drive/1tfIWJ-hKIvXr6vagIkaNN1DhLYmE8NUy)

##### Results on 10000 documents
![result](/results/sample_runs/sample4plot.png)
