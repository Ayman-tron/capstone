import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate(
    "live-b1071-firebase-adminsdk-m1rco-f2f91025a2.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# Generate dummy vibration data and write it to Firestore
status = 'hello Ayman'
doc_ref = db.collection(u'Flange_status').document()
doc_ref.set({
        u'status': status,
})