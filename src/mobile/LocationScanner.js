import React, { useState } from 'react';
import { View, Text, Button, Alert, StyleSheet, Platform, PermissionsAndroid } from 'react-native';
import Geolocation from 'react-native-geolocation-service';

const BACKEND_URL = "http://localhost:8000/verify-delivery"; // Replace with your computer's local IP block (e.g., 192.168.1.x) during mobile testing

export default function LocationScanner({ deviceId, imageHash }) {
  const [location, setLocation] = useState(null);
  const [status, setStatus] = useState("Idle");

  // Request Android Fine Location Permissions
  const requestLocationPermission = async () => {
    if (Platform.OS === 'ios') {
      const auth = await Geolocation.requestAuthorization('whenInUse');
      return auth === 'granted';
    }
    
    if (Platform.OS === 'android') {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
        {
          title: "High-Accuracy Location Permission",
          message: "We need access to your raw location to prevent GPS spoofing and ensure the delivery reaches the correct Humanitarian Node.",
          buttonPositive: "OK"
        }
      );
      return granted === PermissionsAndroid.RESULTS.GRANTED;
    }
  };

  const captureAndSendLocation = async () => {
    const hasPermission = await requestLocationPermission();
    if (!hasPermission) {
      Alert.alert("Permission Denied", "Location is required to verify delivery.");
      return;
    }

    setStatus("Extracting Raw GPS coordinates...");

    Geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        // In a full NMEA extraction scenario, we'd hook into Android's LocationManager.addNmeaListener() via a native Java module.
        // For the React Native side, we secure the highest accuracy coordinates available.
        setLocation({ latitude, longitude });
        sendToBackend(latitude, longitude);
      },
      (error) => {
        Alert.alert("GPS Error", error.message);
        setStatus("Error extracting location");
      },
      { 
        enableHighAccuracy: true, 
        timeout: 15000, 
        maximumAge: 0 // Prevent caching of spoofed locations
      }
    );
  };

  const sendToBackend = async (lat, lon) => {
    setStatus("Verifying Distance with Spatial Oracle backend...");
    try {
      const payload = {
        device_id: deviceId || "device_alpha_1",
        latitude: lat,
        longitude: lon,
        timestamp: new Date().toISOString(),
        image_hash: imageHash || "mock_hash_abc123" // Tied to the MobileNetV2 classified donation photo
      };

      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      
      if (response.ok) {
        setStatus(`Success: ${data.message} (Distance: ${data.distance_km.toFixed(2)}km)`);
        Alert.alert("Delivery Verified", "You are exactly at the NGO warehouse. Event Queued in Redis!");
      } else {
        setStatus(`Blocked: ${data.detail}`);
        Alert.alert("Spoofing Detected or Off-site", data.detail);
      }
    } catch (err) {
      setStatus("Network Error");
      Alert.alert("Error", "Could not connect to FastAPI backend. Ensure uvicorn is running and IP is correct.");
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Delivery Node Verification</Text>
      <Text style={styles.status}>Status: {status}</Text>
      
      {location && (
        <Text style={styles.coords}>
          Lat: {location.latitude.toFixed(6)}{'\n'}Lon: {location.longitude.toFixed(6)}
        </Text>
      )}

      <Button title="Capture & Verify Delivery" onPress={captureAndSendLocation} color="#0052cc" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, alignItems: 'center', justifyContent: 'center', flex: 1, backgroundColor: '#f5f5f5' },
  title: { fontSize: 22, fontWeight: 'bold', marginBottom: 20, textAlign: 'center' },
  status: { fontSize: 16, marginBottom: 15, color: '#0052cc', fontStyle: 'italic', textAlign: 'center' },
  coords: { fontSize: 16, marginBottom: 25, color: '#333', textAlign: 'center', fontWeight: 'bold' }
});
