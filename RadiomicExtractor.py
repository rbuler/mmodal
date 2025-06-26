import time
import logging
import multiprocessing
import SimpleITK as sitk
from tqdm import tqdm #  , trange
from multiprocessing import Pool
from radiomics import featureextractor
logger = logging.getLogger(__name__)


class RadiomicsExtractor():

    def  __init__(self, param_file: str):
        self.extractor = featureextractor.RadiomicsFeatureExtractor(param_file)

    def get_enabled_image_types(self):
        return list(self.extractor.enabledImagetypes.keys())
    
    def get_enabled_features(self):
        return list(self.extractor.enabledFeatures.keys())

    def extract_radiomics(self, list_of_dicts):

        label = self.extractor.settings.get('label', None)
        image = list_of_dicts['image']
        mask = list_of_dicts['mask']

        sitk_image = sitk.GetImageFromArray(image)
        sitk_mask = sitk.GetImageFromArray(mask)
        sitk_image.SetSpacing((1.0, 1.0, 1.0))
        sitk_mask.SetSpacing((1.0, 1.0, 1.0))

        features = self.extractor.execute(sitk_image, sitk_mask, label=label)
        return features
    

    def parallell_extraction(self, list_of_dicts: list, n_processes = None):
        logger.info("Extraction mode: parallel")
        if n_processes is None:
            n_processes = multiprocessing.cpu_count() - 1
        start_time = time.time()
        with Pool(n_processes) as pool:
            results = list(tqdm(pool.imap(self.extract_radiomics, list_of_dicts),
                                 total=len(list_of_dicts)))
        end_time = time.time()

        h, m, s = self._convert_time(start_time, end_time)
        logger.info(f" Time taken: {h}h:{m}m:{s}s")

        return results
    

    def serial_extraction(self, list_of_dicts: list):
        logger.info("Extraction mode: serial")
        all_results = []
            # for item in trange(len(train_df)):
        start_time = time.time()
        for item in range(len(list_of_dicts)):
            all_results.append(self.extract_radiomics(list_of_dicts[item]))
        end_time = time.time()

        h, m, s = self._convert_time(start_time, end_time)
        logger.info(f" Time taken: {h}h:{m}m:{s}s")
        return all_results


    def _convert_time(self, start_time, end_time):
        '''
        Converts time in seconds to hours, minutes and seconds.
        '''
        dt = end_time - start_time
        h, m, s = int(dt // 3600), int((dt % 3600 ) // 60), int(dt % 60)
        return h, m, s
