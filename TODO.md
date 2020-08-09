# To Do

## Base Forecasters

### Base Class
- [ ] Create base class? Is this needed?

### Robust Exponential Smoother 
- [x] Port Robust Existing Exponential Smoother   
- [x] Benchmark Existing Robust Exponential Smoother  
- [x] Migrate Robust Exponential Smoother to Cython  
- [x] Benchmark New Robust Exponential Smoother -> 50x speedup
- [ ] Comment Code

### Exponential Smoother 
- [x] Make Exponential Smoother (non-robust)  
- [x] Benchmark Exponential Smoother vs Robust (2x speedup, but not robust to outliers)

### Facebook Prophet
- [ ] Investigate if this can be used as a base forecaster

## Group Forecasters

### Maximum Likelihood Forecaster
- [ ] Port Existing MLE group forecaster (in progress)   
- [x] Port Consolidation Function
- [ ] Create timeseries class (in progress)
- [ ] Add method for creating s matrix to timeseries class

### New Ideas
- [ ] Gradient Boosting?
- [ ] Random Exponential Smoothing?