/*******************************************************************************
 * Copyright (c) 2019 UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v1.0 which accompanies this
 * distribution. The Eclipse Public License is available at
 * http://www.eclipse.org/legal/epl-v10.html and the Eclipse Distribution
 *License is available at https://eclipse.org/org/documents/edl-v10.php
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
  *   Daniel Claudino - Update to Qiskit Runtime (Qiskit 1.x)
 *******************************************************************************/
#ifndef QUANTUM_GATE_ACCELERATORS_IBMACCELERATOR_HPP_
#define QUANTUM_GATE_ACCELERATORS_IBMACCELERATOR_HPP_

#include "InstructionIterator.hpp"
#include "Accelerator.hpp"
#include <bitset>
#include <type_traits>
#include "IRTransformation.hpp"
#include "Json.hpp"

using namespace xacc;
using json = nlohmann::json;

namespace xacc {
namespace quantum {

class RestClient {

protected:
  bool verbose = false;

public:
  void setVerbose(bool v) { verbose = v; }

  virtual const std::string post(const std::string &remoteUrl,
                                 const std::string &path,
                                 const std::string &postStr,
                                 std::map<std::string, std::string> headers =
                                     std::map<std::string, std::string>{},
                                     const std::string& queryParams = "");

  virtual void put(const std::string &remoteUrl, const std::string &putStr,
                   std::map<std::string, std::string> headers =
                       std::map<std::string, std::string>{});\

  virtual const std::string
  get(const std::string &remoteUrl, const std::string &path,
      std::map<std::string, std::string> headers =
          std::map<std::string, std::string>{},
      std::map<std::string, std::string> extraParams = {});

  virtual ~RestClient() {}
};

#define IS_INTEGRAL(T)                                                         \
  typename std::enable_if<std::is_integral<T>::value>::type * = 0

template <class T>
std::string integral_to_binary_string(T byte, IS_INTEGRAL(T)) {
  std::bitset<sizeof(T) * 8> bs(byte);
  return bs.to_string();
}

std::string hex_string_to_binary_string(std::string hex);

class IBMAccelerator : public Accelerator {
public:
  void cancel() override;
  void retrieve(const std::string jobId, std::shared_ptr<xacc::AcceleratorBuffer> buffer) override;

  std::map<std::string, std::map<int, int>> name2QubitMap;

  void initialize(const HeterogeneousMap &params = {}) override;

  void updateConfiguration(const HeterogeneousMap &config) override;

  const std::vector<std::string> configurationKeys() override {
    return {"shots", "backend", "n-qubits", "check-jobs-limit", "http-verbose", "mode"};
  }

  HeterogeneousMap getProperties() override;

  const std::string getSignature() override {
    return "ibm:" + chosenBackend["backend_name"].get<std::string>();
  }

  std::vector<std::pair<int, int>> getConnectivity() override;

  std::string getNativeCode(std::shared_ptr<CompositeInstruction> program,
                            const HeterogeneousMap &config) override;
  // Return the name of an IRTransformation of type Placement that is
  // preferred for this Accelerator
  const std::string defaultPlacementTransformation() override {
    return "swap-shortest-path";
  }

  const std::string name() const override { return "ibm"; }
  const std::string description() const override {
    return "The IBM Accelerator interacts with the remote IBM "
           "Quantum Experience to launch XACC quantum kernels.";
  }

  void execute(std::shared_ptr<AcceleratorBuffer> buffer,
               const std::shared_ptr<CompositeInstruction> circuit) override;

  void execute(std::shared_ptr<AcceleratorBuffer> buffer,
               const std::vector<std::shared_ptr<CompositeInstruction>>
                   circuits) override;

  bool isRemote() override { return true; }

  IBMAccelerator()
      : Accelerator(), restClient(std::make_shared<RestClient>()) {}

  virtual ~IBMAccelerator() {}

private:

  void searchAPIKey(std::string &key, std::string &hub, std::string &group,
                    std::string &project);

  void findApiKeyInFile(std::string &key, std::string &hub, std::string &group,
                        std::string &project, const std::string &p);

  void selectBackend(std::vector<std::string>& all_available_backends);

  void processBackendCandidate(const nlohmann::json& b);

  bool verifyJobsLimit(std::string& curr_backend);

  std::string post(const std::string &_url, const std::string &path,
                   const std::string &postStr,
                   std::map<std::string, std::string> headers = {},
                   const std::string& queryParams = "");

  void put(const std::string &_url, const std::string &postStr,
           std::map<std::string, std::string> headers = {});

  std::string get(const std::string &_url, const std::string &path,
                  std::map<std::string, std::string> headers =
                      std::map<std::string, std::string>{},
                  std::map<std::string, std::string> extraParams = {});

  std::shared_ptr<RestClient> restClient;

  static const std::string IBM_API_URL;
  static const std::string IBM_TRANSPILER_URL;
  bool useCloudTranspiler = true;

  std::string IBM_CREDENTIALS_PATH = "";

  std::string currentApiToken;
  std::string hub;
  std::string group;
  std::string project;
  std::string backend;


  std::string primitiveId = "sampler";
  std::string ddType;
  std::vector<std::string> DDTYPES{"XX", "XpXm", "XY4"};
  bool enableDD = false;

  int shots = 1024;
  int backendQueueLength = -1; 

  bool jobIsRunning = false;
  std::string currentJobId = "";

  std::vector<std::string> availableBackends;
  nlohmann::json chosenBackend;
  bool multi_meas_enabled = false;
  bool initialized = false;
  nlohmann::json backends_root;
  std::map<std::string, nlohmann::json> backendProperties;
  std::string getBackendPropsResponse = "{}";
  std::string defaults_response = "{}";
  std::string mode = "qasm";
  int requested_n_qubits = 0;
  bool filterByJobsLimit = false;

  std::map<std::string, std::string> headers;

};

} // namespace quantum
} // namespace xacc

#endif
