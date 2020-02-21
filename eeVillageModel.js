// convert to image collection then to single raster
var marolamboCol = ee.ImageCollection([marolambo1, marolambo2, marolambo3, marolambo4, marolambo5, marolambo6, marolambo7, marolambo8]);
var marolambo = marolamboCol.median();

// add ndvi band
var ndvi = marolambo.normalizedDifference(['b4', 'b1']).rename('NDVI');
marolambo = marolambo.addBands(ndvi);

// add sentinel ndbi band from same date image
var sentinelColl = ee.ImageCollection('COPERNICUS/S2')
                 .filterBounds(mangoroRiver)
                 .filterDate('2018-06-04', '2018-06-12') // +/- 4 days from planet
                 .sort('CLOUDY_PIXEL_PERCENTAGE', true);
var sentinel = sentinelColl.first().updateMask(marolambo.select('b1')); // clip to planet image bounds
var ndbi = sentinel.normalizedDifference(['B12', 'B8']).rename('NDBI');
marolambo = marolambo.addBands(ndbi);

// add blue dissimilarity band from gray level co-occurence matrix (5px neighborhood)
var glcm = marolambo.select('b1').int32().glcmTexture(5);
var glcmDissimilarity = glcm.select('b1_diss').rename('diss');
var kernel7 = ee.Kernel.square({
  radius: 5,
  units: 'pixels'
});
var meanDissimilarity = glcmDissimilarity.reduceNeighborhood({
  reducer: ee.Reducer.mean(),
  kernel: kernel7
}).rename('mean_diss');
marolambo = marolambo.addBands(glcmDissimilarity);

// assess separability of land cover classes (mean, median, stdDev)
var coverTypes = ee.FeatureCollection([villages, sandbank, soil, water, ag, forest, shade]);
var coverMeans = marolambo.reduceRegions({
  collection: coverTypes,
  reducer: ee.Reducer.mean(),
  scale: 3
});
var coverMedians = marolambo.reduceRegions({
  collection: coverTypes,
  reducer: ee.Reducer.median(),
  scale: 3
});
var coverStdDev = marolambo.reduceRegions({
  collection: coverTypes,
  reducer: ee.Reducer.stdDev(),
  scale: 3
});

// print separability results to console
print('mean for each cover:');
print(coverMeans);
print('median for each cover:');
print(coverMedians);
print('standard deviation for each cover:');
print(coverStdDev);

// hard supervised classification - random forest classification
var training = marolambo.sampleRegions({
  collection: coverTypes,
  properties: ['cover'],
  scale:      3
});
var classifier = ee.Classifier.randomForest(5).train(training, 'cover');
var classified = marolambo.classify(classifier);

// low pass with 7px kernel (alternative for focal min/max, allows to fill in gaps within patches identified as villages)
var kernel7 = ee.Kernel.circle({
  radius: 7,
  units: 'pixels',
});
var meanConvolution = classified.reduceNeighborhood({
  reducer: ee.Reducer.mean(),
  kernel: kernel7
});
var convDiv = meanConvolution.lt(0.25);
var convMasked = meanConvolution.updateMask(meanConvolution.lt(0.25));

// remove noise with 400px minimum mapping unit (I adjusted this each time I ran the script - 100px, 150px, 200px, 250px, 300px, 350px, 400px)
var mmuPatches = convDiv.connectedPixelCount(100, true);
var finalRaster = convMasked.updateMask(mmuPatches.eq(100));

// export output raster to assets
Export.image.toAsset({
  image: finalRaster,
  description: 'finalRaster7_100mmu',
  assetId: 'DIPAproject/export7_100mmu',
  scale: 3,
  maxPixels: 1000000000000
});

// add layers to map
Map.addLayer(marolambo, {bands: ['b3', 'b2', 'b1'], min: 0, max: 2000}, 'Mangoro Valley: 06/08/2018 (Planet)');
Map.addLayer(classified, {min: 0, max: 6, palette: ['black', 'yellow', 'brown', 'blue', 'green', '#00c50f', '#00c50f']}, 'Random Forrest Classification');
Map.addLayer(finalRaster, {min: 0, max: 6, palette:["ff0000","ffffff","ffffff","ffffff","ffffff","ffffff","ffffff"]}, 'Village Model');
