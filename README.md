# TradeBor
Trade pattern recognition using Smart Money Concept


# Repo Walkthrough
- crawl data using [Chainlink Crawler](./src/dataCrawler/fetchHPChainlink.py) or others media
- create maker to test the visualization at [ICT marker](./src/dataProcessor/1createMarker.ipynb)
- merge file using [merge](./src/dataProcessor/2mergeData.ipynb)
- render it, test the marker on [webserver](./src/webServer/web.py)
- verify the marker location
- or move the merged file to other location

```bash
bash bash/symlink_data.sh
```


