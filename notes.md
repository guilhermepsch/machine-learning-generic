# Documentação

## 1. Processamento de Dados

### Análise do Gráfico Gerado por Síndrome

De acordo com o gráfico gerado em imagens por sindrome, é possivel notar que de fato existem duas sindromes que possuem muito mais imagens que o resto das outras do dataset.

É claro que os dados poderiam estar melhor balanceados, nao acredito que isso cause um problema tão grande visto que os dois são os unicos dentre os 8 que estao mais normalizados.

## 2. Visualização de Dados

### Análise do Gráfico de dispersão t-Sne

É possível observar que as imagens da mesma síndrome tendem a ficar agrupadas próximas umas das outras. No entanto, existem outliers em todas as síndromes. Algumas imagens específicas estão posicionadas no lado oposto do gráfico, o que pode sugerir uma possível similaridade entre síndromes. Acredito que síndromes com prefixos semelhantes possam também ser mais próximas em termos de embeddings. Por exemplo, as síndromes 300000007 e 300000018 podem ser variações de uma mesma síndrome.

Uma evidência que poderia apoiar essa teoria é o fato de que os clusters de prefixos similares estão adjacentes entre si. No entanto, essa observação não é conclusiva, pois não há contexto suficiente sobre os dados para garantir essa relação.

O cluster mais caótico é o relacionado a sindrome 700018215, que coincidentemente é aquele que possui uma das menores presenças no dataset.

No geral, o gráfico t-SNE conseguiu realizar a clusterização de forma eficaz, alocando a maioria dos dados corretamente em seus devidos clusters.

## 3. Classificação
