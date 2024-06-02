This model is in addition to the paper "A global analysis of ice phenology for 3702 lakes and 1028 reservoirs accross the Northern Hemisphere using Sentinel-2". The paper is currently in revision for publication in the journal Cold Regions Science and Technology. 

The random forest model is a sequenced composed of a classifier followed by a regressor, calibrated based on Sentinel-2 observations in order to predict ice formation and the corresponding dates (freeze-up and break-up).

Calibration data are retrieved in the database "all_phenology.csv", which contains lakes and reservoirs physical and climatic characteristics used as input features to the model. These characteristics include latitude, longitude, elevation, area, mean depth, residence time, maximum accumulation of freezing degree days and the corresponding date, total shortwave radiation, annual solid precipitation and mixed layer depth. 

