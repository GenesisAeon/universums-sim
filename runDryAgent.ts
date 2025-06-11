import { AgentConfig, simulateAgent } from './agent-core';

export function runDryAgent(agentConfig: AgentConfig) {
  const log = [];
  try {
    const result = simulateAgent(agentConfig);
    log.push(`Agent ${agentConfig.name} passed dry-check.`);
  } catch (e) {
    log.push(`❌ Agent ${agentConfig.name} failed: ${e.message}`);
  }
  return log;
}
