# iCard Platform Service — API Documentation

**Base URL:** `https://<your-app>.azurewebsites.net`
**Auth:** All endpoints require `Authorization: Bearer <token>` header.

---

## Authentication

```
POST /api/auth/login
```

**Body:**
```json
{
  "authKey": "string",
  "keyValue": "string"
}
```

**Response:**
```json
{
  "token": "eyJ..."
}
```

Use the returned token in the `Authorization: Bearer` header for all subsequent requests.

---

## Warehouses

### List Warehouses
```
GET /api/warehouse?page=1&pageSize=20
```

**Response:**
```json
{
  "items": [
    {
      "warehouseId": 2,
      "entityId": 5,
      "entityName": "Main Entity",
      "name": "Warehouse A",
      "createDate": "2025-01-01T00:00:00Z",
      "whType": 1
    }
  ],
  "totalCount": 10,
  "page": 1,
  "pageSize": 20
}
```

---

### Get Warehouse
```
GET /api/warehouse/{id}
```

**Response:** Single warehouse object (same shape as above).

---

### Create Warehouse
```
POST /api/warehouse
```

**Body:**
```json
{
  "name": "New Warehouse",
  "entityId": 5,
  "whType": 1
}
```

---

### Update Warehouse
```
PUT /api/warehouse/{id}
```

**Body:** Same fields as create, all optional.

---

### Get Warehouse Items
```
GET /api/warehouse/{id}/items
```

Returns all items configured in this warehouse with their current shelf status.

**Response:**
```json
[
  {
    "warehouseId": 2,
    "itemId": 1133,
    "itemName": "المدار 3",
    "faceValue": 3.00,
    "unitName": "JOD",
    "brandName": "المدار",
    "brandRegionName": "Jordan",
    "supplyLimit": 5,
    "shelfSize": 50,
    "supplyMethod": 1,
    "mainProviderAccount": 1,
    "currentShelfCount": 40,
    "needsRestock": false
  }
]
```

| Field | Description |
|---|---|
| `supplyMethod` | `0` = Manual only, `1` = Auto, `2` = Both |
| `shelfSize` | Target stock level |
| `currentShelfCount` | Vouchers currently available to sell |
| `needsRestock` | `true` when count falls below `supplyLimit` |

---

### Get Item Shelf Status
```
GET /api/warehouse/{id}/items/{itemId}/shelf-status
```

**Response:**
```json
{
  "warehouseId": 2,
  "itemId": 1133,
  "shelfCount": 40,
  "shelfSize": 50,
  "supplyLimit": 5,
  "needsRestock": false,
  "inventoryCount": 120
}
```

---

### Get Item Config
```
GET /api/warehouse/{id}/items/{itemId}/config
```

**Response:** Same shape as a single item from Get Warehouse Items.

---

### Update Item Config
```
PUT /api/warehouse/{id}/items/{itemId}/config
```

**Body:**
```json
{
  "shelfSize": 50,
  "supplyLimit": 5,
  "supplyMethod": 1,
  "mainProviderAccount": 1
}
```

All fields are optional. Only provided fields will be updated.

---

### Trigger Restock — Single Item
```
POST /api/warehouse/{id}/items/{itemId}/restock?qty=10
```

- `qty` is optional. If omitted, the system auto-calculates `shelfSize - currentShelfCount`.

**Response:**
```json
{
  "itemId": 1133,
  "success": true,
  "fulfilledQty": 10,
  "supplyOrderId": 4521,
  "message": "Supply completed successfully"
}
```

---

### Trigger Restock — Bulk (Multiple Items)
```
POST /api/warehouse/{id}/restock
```

Restocks one or more items in a single call. Each item runs independently — a failure in one does not stop the others.

**Body:**
```json
{
  "items": [
    { "itemId": 1133, "qty": 10 },
    { "itemId": 1135 }
  ]
}
```

- `qty` is optional per item. If omitted, auto-calculated from shelf config.

**Response:**
```json
[
  {
    "itemId": 1133,
    "success": true,
    "fulfilledQty": 10,
    "supplyOrderId": 4521,
    "message": "Supply completed successfully"
  },
  {
    "itemId": 1135,
    "success": true,
    "fulfilledQty": 8,
    "supplyOrderId": 4522,
    "message": "Supply completed successfully"
  }
]
```

---

### Upload Vouchers (Manual Provider — CSV)
```
POST /api/warehouse/{id}/items/{itemId}/upload-vouchers
Content-Type: multipart/form-data
```

| Field | Type | Description |
|---|---|---|
| `file` | File | CSV file. Columns: `ItemID, SN, SecCode, ExpDate, ExtraData` |

**Response:**
```json
{
  "itemId": 1133,
  "success": true,
  "fulfilledQty": 25,
  "supplyOrderId": 4523,
  "message": "Manual supply completed"
}
```

---

## Provider Accounts

### List Provider Accounts
```
GET /api/provideraccount?page=1&pageSize=20
```

**Response:**
```json
{
  "items": [
    {
      "providerAccountId": 1,
      "providerType": 1,
      "providerTypeName": "OldICard",
      "name": "SBI Card Provider",
      "baseUrl": "https://oldicardprovider.azurewebsites.net",
      "isActive": true,
      "configData": null
    }
  ],
  "totalCount": 1,
  "page": 1,
  "pageSize": 20
}
```

| `providerType` | Name |
|---|---|
| `0` | Manual |
| `1` | OldICard (SBI Card) |

---

### Get Provider Account
```
GET /api/provideraccount/{id}
```

**Response:** Single provider account object (same shape as above).

---

### Create Provider Account
```
POST /api/provideraccount
```

**Body:**
```json
{
  "name": "My Provider",
  "providerType": 1,
  "baseUrl": "https://provider.example.com",
  "configData": "{}"
}
```

---

### Update Provider Account
```
PUT /api/provideraccount/{id}
```

**Body:** Same fields as create, all optional.

---

### Deactivate Provider Account
```
DELETE /api/provideraccount/{id}
```

**Response:** `204 No Content`

---

### Get Provider Catalog
```
GET /api/provideraccount/{id}/catalog
```

Fetches the live item catalog from the provider. Use this to display available provider items so the user can map them to platform items.

**Response:**
```json
[
  {
    "providerItemId": "16",
    "name": "المدار 3",
    "price": 3.50,
    "mappedPlatformItemId": 1133,
    "details": "{\"categoryId\":16,\"cardName\":\"المدار 3\"}"
  },
  {
    "providerItemId": "18",
    "name": "LTT 30",
    "price": 30.00,
    "mappedPlatformItemId": null,
    "details": null
  }
]
```

| Field | Description |
|---|---|
| `providerItemId` | The provider's own ID for this item |
| `mappedPlatformItemId` | The platform `itemId` this is linked to. `null` = not yet mapped |

---

### Get Live Provider Stock
```
GET /api/provideraccount/{id}/stock
```

Returns real-time stock quantities from the provider, joined with platform item IDs where a mapping exists.

**Response:**
```json
[
  {
    "providerItemId": 16,
    "availableQty": 6406,
    "platformItemId": 1133,
    "price": 3.50
  },
  {
    "providerItemId": 18,
    "availableQty": 6511,
    "platformItemId": 1135,
    "price": 30.00
  },
  {
    "providerItemId": 15,
    "availableQty": 0,
    "platformItemId": null,
    "price": null
  }
]
```

- `platformItemId: null` means no mapping has been set up yet for that provider item.
- Results are ordered by `availableQty` descending.

---

### Save Item Mappings
```
POST /api/provideraccount/{id}/mappings
```

Links platform items to provider items. This tells the system which provider item to purchase when restocking a platform item. You can submit one or many mappings in a single call.

**Body:**
```json
[
  {
    "platformItemId": 1133,
    "providerItemId": "16",
    "price": 3.50,
    "details": null
  },
  {
    "platformItemId": 1135,
    "providerItemId": "18",
    "price": 30.00,
    "details": null
  }
]
```

| Field | Required | Description |
|---|---|---|
| `platformItemId` | yes | The platform's item ID |
| `providerItemId` | yes | The provider's item ID (from catalog) |
| `price` | optional | Override price for this mapping |
| `details` | optional | Additional metadata |

**Response:**
```json
{ "message": "2 mapping(s) saved successfully" }
```

Mappings are upserted — re-submitting an existing `platformItemId` updates it instead of creating a duplicate.

---

## Typical Frontend Workflow — Provider Setup

```
1.  GET  /api/provideraccount                  → pick a provider account (e.g. id=1)
2.  GET  /api/provideraccount/1/catalog        → show provider items to the user
3.  GET  /api/item                             → show platform items alongside
4.  (user maps provider items to platform items in the UI)
5.  POST /api/provideraccount/1/mappings       → save the mappings
6.  GET  /api/provideraccount/1/stock          → verify live stock with mapping info
7.  POST /api/warehouse/2/restock              → trigger restock for mapped items
```

---

## Common Error Responses

| Status | Meaning |
|---|---|
| `400` | Bad request — validation error, check `message` field |
| `401` | Missing or invalid token |
| `403` | Token valid but insufficient permissions |
| `404` | Resource not found |
| `500` | Server error — contact backend team |
