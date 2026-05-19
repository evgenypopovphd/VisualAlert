# Фрагмент исходного кода программы VisualAlert
import numpy as np
from sklearn.ensemble import IsolationForest
 
 
class VisualAlert:
   def __init__(self, reference_profile):
       self.reference_profile = reference_profile
       self.alerts = []
 
       self.detector = IsolationForest(
           contamination=0.05,
           random_state=42
       )
 
   def train_reference_model(self, historical_data):
       self.detector.fit(historical_data)
 
   def detect_anomaly(self, sensor_data):
       prediction = self.detector.predict(
           [sensor_data]
       )
 
       return prediction[0] == -1
 
   def generate_alert(self, sensor_data):
       deviation = np.abs(
           sensor_data - self.reference_profile
       )
 
       alert = {
           "status": "warning",
           "deviation_score": float(
               np.mean(deviation)
           ),
           "sensor_values": sensor_data.tolist()
       }
 
       self.alerts.append(alert)
 
       return alert
 
   def process_stream(self, sensor_stream):
       results = []
 
       for sensor_data in sensor_stream:
 
           if self.detect_anomaly(sensor_data):
               alert = self.generate_alert(
                   sensor_data
               )
 
               results.append(alert)
 
       return results
 
 
if __name__ == "__main__":
 
   reference_profile = np.array(
       [50, 75, 90, 120]
   )
 
   historical_data = np.random.normal(
       loc=reference_profile,
       scale=5,
       size=(100, 4)
   )
 
   sensor_stream = [
       np.random.normal(
           loc=reference_profile,
           scale=5,
           size=4
       )
       for _ in range(10)
   ]
 
   sensor_stream.append(
       np.array([120, 180, 30, 250])
   )
 
   visual_alert = VisualAlert(
       reference_profile
   )
 
   visual_alert.train_reference_model(
       historical_data
   )
 
   alerts = visual_alert.process_stream(
       sensor_stream
   )
 
   print(
       f"Generated {len(alerts)} alerts"
   )