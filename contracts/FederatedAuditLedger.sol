// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FederatedAuditLedger
 * @notice Permissioned Consortium Smart Contract for Blockchain-Enabled Federated AI
 * Governs participant registration, cryptographic weight commitment logging,
 * Byzantine anomaly challenge/slashing, and verifiable global model audit trails.
 * Solves the Ramanan et al. (BAFFLE) gas bottleneck via Hybrid On-Chain Commitment.
 */
contract FederatedAuditLedger {
    // Consortium Governance Admin
    address public consortiumAdmin;

    enum ParticipantStatus { Pending, Active, Suspended, Slashed }

    struct Participant {
        string name;
        string organizationType; // Healthcare, Financial, IoT Gateway
        string publicKeyPem;
        ParticipantStatus status;
        uint256 registrationTimestamp;
        uint256 totalAcceptedRounds;
        uint256 totalSlashedRounds;
    }

    struct ClientCommitment {
        address clientAddress;
        string weightHash; // SHA-256 fingerprint of model weights
        bytes signature;   // ECDSA cryptographic signature
        uint32 sampleCount;
        bool isAccepted;
        uint256 submissionTimestamp;
    }

    struct RoundAuditRecord {
        uint256 roundNumber;
        string previousGlobalModelHash;
        string updatedGlobalModelHash;
        string merkleRoot;
        uint16 accuracyBps; // Basis points: 8950 = 89.50%
        uint16 lossBps;     // Basis points: 125 = 0.0125
        uint256 completedTimestamp;
        uint256 participantCount;
        uint256 slashedCount;
        bool isFinalized;
    }

    // Mappings
    mapping(address => Participant) public participants;
    address[] public participantAddresses;

    // Round number => Round audit record
    mapping(uint256 => RoundAuditRecord) public rounds;
    // Round number => List of client commitments
    mapping(uint256 => ClientCommitment[]) public roundCommitments;
    // Round number => client address => has submitted
    mapping(uint256 => mapping(address => bool)) public hasCommitted;

    uint256 public currentRound;
    uint256 public totalRoundsCompleted;

    // Events for real-time monitoring and SOC forensics
    event ParticipantRegistered(address indexed clientAddress, string name, string organizationType);
    event ParticipantStatusUpdated(address indexed clientAddress, ParticipantStatus status);
    event RoundStarted(uint256 indexed roundNumber, string previousGlobalModelHash, uint256 timestamp);
    event CommitmentSubmitted(uint256 indexed roundNumber, address indexed clientAddress, string weightHash);
    event ByzantineClientSlashed(uint256 indexed roundNumber, address indexed clientAddress, string reason);
    event RoundFinalized(
        uint256 indexed roundNumber,
        string updatedGlobalModelHash,
        string merkleRoot,
        uint16 accuracyBps,
        uint16 lossBps,
        uint256 acceptedParticipants
    );

    modifier onlyAdmin() {
        require(msg.sender == consortiumAdmin, "Caller is not consortium admin");
        _;
    }

    modifier onlyActiveParticipant() {
        require(participants[msg.sender].status == ParticipantStatus.Active, "Participant is not active");
        _;
    }

    constructor() {
        consortiumAdmin = msg.sender;
        currentRound = 0;
    }

    /**
     * @notice Registers an authorized enterprise node in the consortium (Hospital, Bank, IoT)
     */
    function registerParticipant(
        address _clientAddress,
        string memory _name,
        string memory _organizationType,
        string memory _publicKeyPem
    ) external onlyAdmin {
        require(participants[_clientAddress].status == ParticipantStatus.Pending, "Participant already exists");

        participants[_clientAddress] = Participant({
            name: _name,
            organizationType: _organizationType,
            publicKeyPem: _publicKeyPem,
            status: ParticipantStatus.Active,
            registrationTimestamp: block.timestamp,
            totalAcceptedRounds: 0,
            totalSlashedRounds: 0
        });

        participantAddresses.push(_clientAddress);
        emit ParticipantRegistered(_clientAddress, _name, _organizationType);
    }

    /**
     * @notice Initiates a new federated intelligence training round
     */
    function startRound(uint256 _roundNumber, string memory _baseModelHash) external onlyAdmin {
        require(_roundNumber == currentRound + 1, "Invalid round progression");
        currentRound = _roundNumber;

        rounds[_roundNumber].roundNumber = _roundNumber;
        rounds[_roundNumber].previousGlobalModelHash = _baseModelHash;
        rounds[_roundNumber].isFinalized = false;

        emit RoundStarted(_roundNumber, _baseModelHash, block.timestamp);
    }

    /**
     * @notice Submits a client's signed weight commitment for the active round
     */
    function submitCommitment(
        uint256 _roundNumber,
        string memory _weightHash,
        bytes memory _signature,
        uint32 _sampleCount
    ) external onlyActiveParticipant {
        require(_roundNumber == currentRound, "Not active round");
        require(!hasCommitted[_roundNumber][msg.sender], "Already submitted commitment");

        hasCommitted[_roundNumber][msg.sender] = true;

        roundCommitments[_roundNumber].push(ClientCommitment({
            clientAddress: msg.sender,
            weightHash: _weightHash,
            signature: _signature,
            sampleCount: _sampleCount,
            isAccepted: true,
            submissionTimestamp: block.timestamp
        }));

        emit CommitmentSubmitted(_roundNumber, msg.sender, _weightHash);
    }

    /**
     * @notice Records Byzantine anomaly detection and slashes malicious client contribution
     */
    function slashMaliciousParticipant(
        uint256 _roundNumber,
        address _clientAddress,
        string memory _reason
    ) external onlyAdmin {
        require(hasCommitted[_roundNumber][_clientAddress], "Client did not submit in this round");

        // Mark commitment as rejected
        ClientCommitment[] storage comms = roundCommitments[_roundNumber];
        for (uint i = 0; i < comms.length; i++) {
            if (comms[i].clientAddress == _clientAddress) {
                comms[i].isAccepted = false;
                break;
            }
        }

        participants[_clientAddress].totalSlashedRounds += 1;
        emit ByzantineClientSlashed(_roundNumber, _clientAddress, _reason);
    }

    /**
     * @notice Finalizes federated aggregation round and publishes cryptographic audit proof
     */
    function finalizeRound(
        uint256 _roundNumber,
        string memory _updatedGlobalModelHash,
        string memory _merkleRoot,
        uint16 _accuracyBps,
        uint16 _lossBps
    ) external onlyAdmin {
        require(_roundNumber == currentRound, "Round mismatch");
        require(!rounds[_roundNumber].isFinalized, "Round already finalized");

        RoundAuditRecord storage r = rounds[_roundNumber];
        r.updatedGlobalModelHash = _updatedGlobalModelHash;
        r.merkleRoot = _merkleRoot;
        r.accuracyBps = _accuracyBps;
        r.lossBps = _lossBps;
        r.completedTimestamp = block.timestamp;
        r.isFinalized = true;

        uint256 accepted = 0;
        uint256 slashed = 0;
        ClientCommitment[] storage comms = roundCommitments[_roundNumber];
        for (uint i = 0; i < comms.length; i++) {
            if (comms[i].isAccepted) {
                accepted++;
                participants[comms[i].clientAddress].totalAcceptedRounds += 1;
            } else {
                slashed++;
            }
        }
        r.participantCount = accepted;
        r.slashedCount = slashed;

        totalRoundsCompleted++;

        emit RoundFinalized(_roundNumber, _updatedGlobalModelHash, _merkleRoot, _accuracyBps, _lossBps, accepted);
    }

    /**
     * @notice Returns total number of registered participants
     */
    function getParticipantCount() external view returns (uint256) {
        return participantAddresses.length;
    }

    /**
     * @notice Fetches commitment details for a given round and index
     */
    function getCommitment(uint256 _roundNumber, uint256 index) external view returns (
        address clientAddress,
        string memory weightHash,
        uint32 sampleCount,
        bool isAccepted,
        uint256 submissionTimestamp
    ) {
        ClientCommitment memory c = roundCommitments[_roundNumber][index];
        return (c.clientAddress, c.weightHash, c.sampleCount, c.isAccepted, c.submissionTimestamp);
    }
}
