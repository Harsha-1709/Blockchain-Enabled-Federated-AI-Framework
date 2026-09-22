"""
Cryptographic Verification Engine
Implements:
- SECP256k1 / SECP256R1 ECDSA key-pair generation
- SHA-256 digital fingerprinting of weight vectors
- Cryptographic digital signing and verification of client payloads
- Merkle Tree computation for transaction batches & weight chunks
Prevents tampering, spoofing, and man-in-the-middle attacks.
"""

import hashlib
import json
import numpy as np
from typing import Dict, Any, List, Tuple
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

def hash_sha256(data: bytes) -> str:
    """Computes standard SHA-256 hex digest of raw bytes"""
    return hashlib.sha256(data).hexdigest()

def hash_weights(weights: np.ndarray) -> str:
    """
    Produces deterministic SHA-256 cryptographic fingerprint of a model weight vector.
    Rounds to float32 precision and serializes bytes.
    """
    arr_bytes = weights.astype(np.float32).tobytes()
    return hash_sha256(arr_bytes)

def compute_merkle_root(leaf_hashes: List[str]) -> str:
    """
    Computes a cryptographic Merkle Root given a list of hex hash strings.
    """
    if not leaf_hashes:
        return hash_sha256(b"empty_tree")
    if len(leaf_hashes) == 1:
        return leaf_hashes[0]

    current_level = [bytes.fromhex(h) for h in leaf_hashes]

    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else left
            combined = hashlib.sha256(left + right).digest()
            next_level.append(combined)
        current_level = next_level

    return current_level[0].hex()

class NodeKeyManager:
    """
    Manages ECDSA asymmetric key pairs for consortium participants
    (Hospital A, Bank B, IoT Node C, Smart Contract Aggregator).
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        # Use SECP256R1 (ECDSA)
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()

    def get_public_key_hex(self) -> str:
        """Returns public key in PEM hex format"""
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.hex()

    def sign_payload(self, message: str) -> str:
        """Signs a text/hash payload with the node's private key; returns signature hex"""
        signature = self.private_key.sign(
            message.encode("utf-8"),
            ec.ECDSA(hashes.SHA256())
        )
        return signature.hex()

    @staticmethod
    def verify_signature(public_key_hex: str, message: str, signature_hex: str) -> bool:
        """Verifies signature against public key hex and message string"""
        try:
            pem_bytes = bytes.fromhex(public_key_hex)
            pub_key = serialization.load_pem_public_key(pem_bytes)
            sig_bytes = bytes.fromhex(signature_hex)
            pub_key.verify(
                sig_bytes,
                message.encode("utf-8"),
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except (InvalidSignature, ValueError, Exception):
            return False
