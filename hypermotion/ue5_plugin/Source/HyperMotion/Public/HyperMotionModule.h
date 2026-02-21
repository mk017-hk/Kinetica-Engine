#pragma once

#include "Modules/ModuleManager.h"

class FHyperMotionModule : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;
};
