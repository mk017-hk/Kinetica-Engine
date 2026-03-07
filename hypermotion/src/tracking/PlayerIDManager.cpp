#include "HyperMotion/tracking/PlayerIDManager.h"
#include "HyperMotion/core/Logger.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace hm::tracking {

static constexpr const char* TAG = "PlayerIDManager";

static float cosineSimilarity(const std::array<float, REID_DIM>& a,
                               const std::array<float, REID_DIM>& b) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < REID_DIM; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 1e-8f ? dot / denom : 0.0f;
}

struct PlayerIDManager::Impl {
    PlayerIDManagerConfig config;
    std::unordered_map<int, PlayerIdentity> identities;  // persistentID -> identity
    int nextID = 0;

    // Map from frame-level tracklet ID to our persistent ID
    std::unordered_map<int, int> trackletToPlayer;
};

PlayerIDManager::PlayerIDManager(const PlayerIDManagerConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

PlayerIDManager::~PlayerIDManager() = default;
PlayerIDManager::PlayerIDManager(PlayerIDManager&&) noexcept = default;
PlayerIDManager& PlayerIDManager::operator=(PlayerIDManager&&) noexcept = default;

std::unordered_map<int, int> PlayerIDManager::update(
    const std::vector<DetectedPerson>& persons, int frameIndex) {

    std::unordered_map<int, int> mapping;

    // Prune identities that haven't been seen for too long
    for (auto it = impl_->identities.begin(); it != impl_->identities.end();) {
        if (frameIndex - it->second.lastSeenFrame > impl_->config.maxLostFrames) {
            impl_->trackletToPlayer.erase(it->second.persistentID);
            it = impl_->identities.erase(it);
        } else {
            ++it;
        }
    }

    for (const auto& person : persons) {
        int trackletID = person.id;

        // Check if this tracklet is already mapped
        auto existingIt = impl_->trackletToPlayer.find(trackletID);
        if (existingIt != impl_->trackletToPlayer.end()) {
            int pid = existingIt->second;
            auto& identity = impl_->identities[pid];
            identity.totalFramesSeen++;
            identity.lastSeenFrame = frameIndex;

            // EMA update of ReID template
            float alpha = impl_->config.reidTemplateAlpha;
            for (int i = 0; i < REID_DIM; ++i) {
                identity.reidTemplate[i] =
                    (1.0f - alpha) * identity.reidTemplate[i] +
                    alpha * person.reidFeature[i];
            }

            mapping[trackletID] = pid;
            continue;
        }

        // Try to re-identify from lost identities using ReID features
        int bestMatchID = -1;
        float bestSim = impl_->config.reidMatchThreshold;

        // Check if ReID feature is non-zero
        float reidNorm = 0;
        for (float v : person.reidFeature) reidNorm += v * v;
        bool hasReid = reidNorm > 1e-6f;

        if (hasReid) {
            for (auto& [pid, identity] : impl_->identities) {
                // Only try to match identities not currently active
                bool alreadyMapped = false;
                for (auto& [tid, mpid] : impl_->trackletToPlayer) {
                    if (mpid == pid) { alreadyMapped = true; break; }
                }
                if (alreadyMapped) continue;

                float sim = cosineSimilarity(person.reidFeature, identity.reidTemplate);
                if (sim > bestSim) {
                    bestSim = sim;
                    bestMatchID = pid;
                }
            }
        }

        if (bestMatchID >= 0) {
            // Re-identified a lost player
            auto& identity = impl_->identities[bestMatchID];
            identity.totalFramesSeen++;
            identity.lastSeenFrame = frameIndex;
            impl_->trackletToPlayer[trackletID] = bestMatchID;
            mapping[trackletID] = bestMatchID;
            HM_LOG_DEBUG(TAG, "Re-identified tracklet " + std::to_string(trackletID) +
                         " as player " + std::to_string(bestMatchID) +
                         " (sim=" + std::to_string(bestSim) + ")");
        } else {
            // Create new identity
            if (static_cast<int>(impl_->identities.size()) >= impl_->config.maxPlayers) {
                HM_LOG_WARN(TAG, "Max players reached, ignoring new tracklet " +
                           std::to_string(trackletID));
                continue;
            }

            int pid = impl_->nextID++;
            PlayerIdentity identity;
            identity.persistentID = pid;
            identity.label = person.classLabel;
            identity.totalFramesSeen = 1;
            identity.firstSeenFrame = frameIndex;
            identity.lastSeenFrame = frameIndex;
            identity.reidTemplate = person.reidFeature;
            impl_->identities[pid] = identity;
            impl_->trackletToPlayer[trackletID] = pid;
            mapping[trackletID] = pid;
        }
    }

    return mapping;
}

std::vector<PlayerIdentity> PlayerIDManager::getActiveIdentities() const {
    std::vector<PlayerIdentity> result;
    result.reserve(impl_->identities.size());
    for (const auto& [id, identity] : impl_->identities) {
        result.push_back(identity);
    }
    return result;
}

const PlayerIdentity* PlayerIDManager::getIdentity(int persistentID) const {
    auto it = impl_->identities.find(persistentID);
    return it != impl_->identities.end() ? &it->second : nullptr;
}

int PlayerIDManager::totalPlayersTracked() const {
    return impl_->nextID;
}

void PlayerIDManager::reset() {
    impl_->identities.clear();
    impl_->trackletToPlayer.clear();
    impl_->nextID = 0;
}

} // namespace hm::tracking
