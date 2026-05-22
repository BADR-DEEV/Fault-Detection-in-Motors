// #include <WiFi.h>
// #include <HTTPClient.h>
// #include <ArduinoJson.h>

// const char* ssid = "YOUR_WIFI_NAME";
// const char* password = "YOUR_WIFI_PASSWORD";

// const char* serverUrl = "http://192.168.1.100:8000/predict"; // change to your PC IP

// #define BUFFER_SIZE 200
// #define THRESHOLD 12.0

// float ax_buffer[BUFFER_SIZE];
// float ay_buffer[BUFFER_SIZE];
// float az_buffer[BUFFER_SIZE];

// int buffer_index = 0;

// float simulateX() { return random(-300,300)/100.0; }
// float simulateY() { return random(-300,300)/100.0; }
// float simulateZ() { return 9.8 + random(-200,200)/100.0; }

// float magnitude(float x, float y, float z)
// {
//     return sqrt(x*x + y*y + z*z);
// }

// void connectWiFi()
// {
//     WiFi.begin(ssid, password);

//     Serial.print("Connecting to WiFi");

//     while (WiFi.status() != WL_CONNECTED)
//     {
//         delay(500);
//         Serial.print(".");
//     }

//     Serial.println("\nConnected!");
// }

// void sendSnapshot()
// {
//     if (WiFi.status() != WL_CONNECTED) return;

//     HTTPClient http;
//     http.begin(serverUrl);
//     http.addHeader("Content-Type", "application/json");

//     StaticJsonDocument<8192> doc;

//     JsonArray ax = doc.createNestedArray("ax");
//     JsonArray ay = doc.createNestedArray("ay");
//     JsonArray az = doc.createNestedArray("az");

//     for(int i=0;i<BUFFER_SIZE;i++)
//     {
//         ax.add(ax_buffer[i]);
//         ay.add(ay_buffer[i]);
//         az.add(az_buffer[i]);
//     }

//     String json;
//     serializeJson(doc, json);

//     int response = http.POST(json);

//     Serial.print("Server response: ");
//     Serial.println(response);

//     if(response > 0)
//     {
//         String body = http.getString();
//         Serial.println(body);
//     }

//     http.end();
// }

// void setup()
// {
//     Serial.begin(115200);

//     connectWiFi();

//     randomSeed(analogRead(0));
// }

// void loop()
// {
//     float ax = simulateX();
//     float ay = simulateY();
//     float az = simulateZ();

//     float mag = magnitude(ax,ay,az);

//     ax_buffer[buffer_index] = ax;
//     ay_buffer[buffer_index] = ay;
//     az_buffer[buffer_index] = az;

//     buffer_index++;

//     if(buffer_index >= BUFFER_SIZE)
//         buffer_index = 0;

//     Serial.print("Magnitude: ");
//     Serial.println(mag);

//     if(mag > THRESHOLD)
//     {
//         Serial.println("Threshold exceeded → sending snapshot");

//         sendSnapshot();

//         delay(3000);
//     }

//     delay(10);
// }'
#include <iostream>

int main() {
    std::cout << "Hello, C++ is working!" << std::endl;

    int version = 17;
    std::cout << "Using C++ version: " << version << std::endl;

    return 0;
}