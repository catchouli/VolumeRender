#include "serialisation/Serialiser.h"

#include <json/json.h>
#include <Box2D/Box2D.h>
#include <map>

#include "VehicleSim.h"
#include "input/MotorInput.h"
#include "tools/tools/CamFollow.h"

namespace vlr
{
	std::string Serialiser::serialiseWorld(const VehicleSim* vehicleSim,
		const b2World* world)
	{
		unsigned int lastBodyId = 0;
		std::map<const b2Body*, unsigned int> bodyIds;

		// Create writer
		Json::StyledWriter writer;

		// Create root
		Json::Value root;

		// Populate root
		root["drawBodies"] = (vehicleSim->_worldRenderer.GetFlags() & b2Draw::e_shapeBit) > 0;
		root["drawJoints"] = (vehicleSim->_worldRenderer.GetFlags() & b2Draw::e_jointBit) > 0;
		root["drawAABBs"] = (vehicleSim->_worldRenderer.GetFlags() & b2Draw::e_aabbBit) > 0;

		root["gravity"]["x"] = (double)world->GetGravity().x;
		root["gravity"]["y"] = (double)world->GetGravity().y;
		
		root["camPos"]["x"] = vehicleSim->_camera.getPos().x;
		root["camPos"]["y"] = vehicleSim->_camera.getPos().y;

		root["camOrthoScale"] = vehicleSim->_orthoScale;

		// Save bodies
		root["bodyCount"] = world->GetBodyCount();
		root["bodies"] = Json::arrayValue;

		for (const b2Body* body = world->GetBodyList(); body; body = body->GetNext())
		{
			// Assign next body id
			unsigned int bodyId = lastBodyId++;
			bodyIds[body] = bodyId;

			// Create body in json
			Json::Value& bodyVal = root["bodies"][bodyId];
			bodyVal["id"] = bodyId;

			// Store body values
			bodyVal["type"] = body->GetType();
			bodyVal["position"]["x"] = body->GetPosition().x;
			bodyVal["position"]["y"] = body->GetPosition().y;
			bodyVal["angle"] = body->GetAngle();
			bodyVal["linearVelocity"]["x"] = body->GetLinearVelocity().x;
			bodyVal["linearVelocity"]["y"] = body->GetLinearVelocity().y;
			bodyVal["angularVelocity"] = body->GetAngularVelocity();
			bodyVal["lineardamping"] = body->GetLinearDamping();
			bodyVal["angularDampening"] = body->GetAngularDamping();
			bodyVal["allowSleep"] = body->IsSleepingAllowed();
			bodyVal["fixedRotation"] = body->IsFixedRotation();
			bodyVal["bullet"] = body->IsBullet();
			bodyVal["active"] = body->IsActive();
			bodyVal["gravityScale"] = body->GetGravityScale();

			// Save fixtures
			int fixtureCount = 0;
			bodyVal["fixtures"] = Json::arrayValue;

			for (const b2Fixture* fixture = body->GetFixtureList(); fixture; fixture = fixture->GetNext())
			{
				const b2Shape* shape = fixture->GetShape();
				
				Json::Value& fixtureVal = bodyVal["fixtures"][fixtureCount];
				fixtureVal["id"] = fixtureCount;

				// Store fixture values
				fixtureVal["friction"] = fixture->GetFriction();
				fixtureVal["restitution"] = fixture->GetRestitution();
				fixtureVal["density"] = fixture->GetDensity();
				fixtureVal["isSensor"] = fixture->IsSensor();

				// Store filtering data
				fixtureVal["filter"]["categoryBits"] = fixture->GetFilterData().categoryBits;
				fixtureVal["filter"]["maskBits"] = fixture->GetFilterData().maskBits;
				fixtureVal["filter"]["groupIndex"] = fixture->GetFilterData().groupIndex;

				// Store shape
				Json::Value& shapeVal = fixtureVal["shape"];
				switch (shape->GetType())
				{
				case b2Shape::e_circle:
					{
						shapeVal["type"] = shape->GetType();

						b2CircleShape* circle = (b2CircleShape*)shape;
						
						shapeVal["radius"] = circle->m_radius;
						shapeVal["position"]["x"] = circle->m_p.x;
						shapeVal["position"]["y"] = circle->m_p.y;
					}
					break;
				case b2Shape::e_polygon:
					{
						shapeVal["type"] = shape->GetType();
						shapeVal["vertices"] = Json::arrayValue;

						b2PolygonShape* poly = (b2PolygonShape*)shape;

						shapeVal["vertexCount"] = poly->GetVertexCount();

						for (int i = 0; i < poly->GetVertexCount(); ++i)
						{
							Json::Value vertex;
							vertex["x"] = poly->GetVertex(i).x;
							vertex["y"] = poly->GetVertex(i).y;

							shapeVal["vertices"].append(vertex);
						}
					}
					break;
				default:
					printf("serialiseWorld: unknown shape type %d\n", shape->GetType());
					break;
				}

				fixtureCount++;
			}

			bodyVal["fixtureCount"] = fixtureCount;
		}

		// Save joints
		int currentJoint = 0;
		root["jointCount"] = world->GetJointCount();
		root["joints"] = Json::arrayValue;

		for (b2Joint* joint = (b2Joint*)world->GetJointList(); joint;
			joint = joint->GetNext())
		{
			// Get current joint value
			Json::Value& jointVal = root["joints"][currentJoint];
			jointVal["id"] = currentJoint;
			
			// Store common attributes
			// Store joint type
			jointVal["jointType"] = joint->GetType();

			jointVal["bodyA"] = bodyIds[joint->GetBodyA()];
			jointVal["bodyB"] = bodyIds[joint->GetBodyB()];

			jointVal["collideConnected"] = joint->GetCollideConnected();

			// Get input info
			if (joint->GetUserData() != nullptr)
			{
				MotorInput* mo = (MotorInput*)joint->GetUserData();
			
				// Store motor input info
				jointVal["mo"]["set"] = true;
				jointVal["mo"]["enabled"] = mo->getEnabled();
				jointVal["mo"]["maxForce"] = mo->getMaxForce();
				jointVal["mo"]["speed"] = mo->getSpeed();
				jointVal["mo"]["forwardButton"] = mo->getForwardKey();
				jointVal["mo"]["reverseButton"] = mo->getReverseKey();
			}
			else
			{
				jointVal["mo"]["set"] = false;
			}

			// Store per-type attributes
			switch (joint->GetType())
			{
			case e_revoluteJoint:
				{
					b2RevoluteJoint* specJoint = (b2RevoluteJoint*)joint;
					
					jointVal["localAnchorA"]["x"] = specJoint->GetLocalAnchorA().x;
					jointVal["localAnchorA"]["y"] = specJoint->GetLocalAnchorA().y;

					jointVal["localAnchorB"]["x"] = specJoint->GetLocalAnchorB().x;
					jointVal["localAnchorB"]["y"] = specJoint->GetLocalAnchorB().y;

					jointVal["referenceAngle"] = specJoint->GetReferenceAngle();

					jointVal["enableLimit"] = specJoint->IsLimitEnabled();

					jointVal["lowerAngle"] = specJoint->GetLowerLimit();
					jointVal["upperAngle"] = specJoint->GetUpperLimit();

					jointVal["enableMotor"] = specJoint->IsMotorEnabled();
					jointVal["motorSpeed"] = specJoint->GetMotorSpeed();
					jointVal["maxMotorTorque"] = specJoint->GetMaxMotorTorque();
				}
				break;
			case e_prismaticJoint:
				{
					b2PrismaticJoint* specJoint = (b2PrismaticJoint*)joint;
					
					jointVal["localAnchorA"]["x"] = specJoint->GetLocalAnchorA().x;
					jointVal["localAnchorA"]["y"] = specJoint->GetLocalAnchorA().y;

					jointVal["localAnchorB"]["x"] = specJoint->GetLocalAnchorB().x;
					jointVal["localAnchorB"]["y"] = specJoint->GetLocalAnchorB().y;

					jointVal["localAxisA"]["x"] = specJoint->GetLocalAxisA().x;
					jointVal["localAxisA"]["y"] = specJoint->GetLocalAxisA().y;

					jointVal["referenceAngle"] = specJoint->GetReferenceAngle();

					jointVal["enableLimit"] = specJoint->IsLimitEnabled();

					jointVal["lowerTranslation"] = specJoint->GetLowerLimit();
					jointVal["upperTranslation"] = specJoint->GetUpperLimit();

					jointVal["enableMotor"] = specJoint->IsMotorEnabled();
					jointVal["motorSpeed"] = specJoint->GetMotorSpeed();
					jointVal["maxMotorForce"] = specJoint->GetMaxMotorForce();
				}
				break;
			case e_distanceJoint:
				{
					b2DistanceJoint* specJoint = (b2DistanceJoint*)joint;
					
					jointVal["localAnchorA"]["x"] = specJoint->GetLocalAnchorA().x;
					jointVal["localAnchorA"]["y"] = specJoint->GetLocalAnchorA().y;

					jointVal["localAnchorB"]["x"] = specJoint->GetLocalAnchorB().x;
					jointVal["localAnchorB"]["y"] = specJoint->GetLocalAnchorB().y;

					jointVal["length"] = specJoint->GetLength();
					
					jointVal["frequency"] = specJoint->GetFrequency();
					jointVal["dampingRatio"] = specJoint->GetDampingRatio();
				}
				break;
			case e_pulleyJoint:
				{
					b2PulleyJoint* specJoint = (b2PulleyJoint*)joint;

					b2Vec2 localAnchorA = specJoint->GetBodyA()->GetLocalPoint(specJoint->GetAnchorA());
					b2Vec2 localAnchorB = specJoint->GetBodyB()->GetLocalPoint(specJoint->GetAnchorB());
					
					jointVal["groundAnchorA"]["x"] = specJoint->GetGroundAnchorA().x;
					jointVal["groundAnchorA"]["y"] = specJoint->GetGroundAnchorA().y;

					jointVal["groundAnchorB"]["x"] = specJoint->GetGroundAnchorB().x;
					jointVal["groundAnchorB"]["y"] = specJoint->GetGroundAnchorB().y;
					
					jointVal["localAnchorA"]["x"] = localAnchorA.x;
					jointVal["localAnchorA"]["y"] = localAnchorA.y;

					jointVal["localAnchorB"]["x"] = localAnchorB.x;
					jointVal["localAnchorB"]["y"] = localAnchorB.y;

					jointVal["lengthA"] = specJoint->GetLengthA();
					jointVal["lengthB"] = specJoint->GetLengthB();

					jointVal["ratio"] = specJoint->GetRatio();
				}
				break;
			case e_mouseJoint:
				{
					b2MouseJoint* specJoint = (b2MouseJoint*)joint;
					
					jointVal["target"]["x"] = specJoint->GetTarget().x;
					jointVal["target"]["y"] = specJoint->GetTarget().y;

					jointVal["maxForce"] = specJoint->GetMaxForce();
					
					jointVal["frequency"] = specJoint->GetFrequency();

					jointVal["dampingRatio"] = specJoint->GetDampingRatio();
				}
				break;
			case e_wheelJoint:
				{
					b2WheelJoint* specJoint = (b2WheelJoint*)joint;
					
					jointVal["localAnchorA"]["x"] = specJoint->GetLocalAnchorA().x;
					jointVal["localAnchorA"]["y"] = specJoint->GetLocalAnchorA().y;

					jointVal["localAnchorB"]["x"] = specJoint->GetLocalAnchorB().x;
					jointVal["localAnchorB"]["y"] = specJoint->GetLocalAnchorB().y;

					jointVal["localAxisA"]["x"] = specJoint->GetLocalAxisA().x;
					jointVal["localAxisA"]["y"] = specJoint->GetLocalAxisA().y;

					jointVal["enableMotor"] = specJoint->IsMotorEnabled();
					jointVal["motorSpeed"] = specJoint->GetMotorSpeed();
					jointVal["maxMotorTorque"] = specJoint->GetMaxMotorTorque();
					
					jointVal["frequency"] = specJoint->GetSpringFrequencyHz();

					jointVal["dampingRatio"] = specJoint->GetSpringDampingRatio();
				}
				break;
			case e_weldJoint:
				{
					b2WeldJoint* specJoint = (b2WeldJoint*)joint;
					
					jointVal["localAnchorA"]["x"] = specJoint->GetLocalAnchorA().x;
					jointVal["localAnchorA"]["y"] = specJoint->GetLocalAnchorA().y;

					jointVal["localAnchorB"]["x"] = specJoint->GetLocalAnchorB().x;
					jointVal["localAnchorB"]["y"] = specJoint->GetLocalAnchorB().y;

					jointVal["referenceAngle"] = specJoint->GetReferenceAngle();
					
					jointVal["frequency"] = specJoint->GetFrequency();

					jointVal["dampingRatio"] = specJoint->GetDampingRatio();					
				}
				break;
			case e_frictionJoint:
				{
					b2FrictionJoint* specJoint = (b2FrictionJoint*)joint;
					
					jointVal["localAnchorA"]["x"] = specJoint->GetLocalAnchorA().x;
					jointVal["localAnchorA"]["y"] = specJoint->GetLocalAnchorA().y;

					jointVal["localAnchorB"]["x"] = specJoint->GetLocalAnchorB().x;
					jointVal["localAnchorB"]["y"] = specJoint->GetLocalAnchorB().y;
					
					jointVal["maxForce"] = specJoint->GetMaxForce();
					jointVal["maxTorque"] = specJoint->GetMaxTorque();
				}
				break;
			case e_ropeJoint:
				{
					b2RopeJoint* specJoint = (b2RopeJoint*)joint;
					
					jointVal["localAnchorA"]["x"] = specJoint->GetLocalAnchorA().x;
					jointVal["localAnchorA"]["y"] = specJoint->GetLocalAnchorA().y;

					jointVal["localAnchorB"]["x"] = specJoint->GetLocalAnchorB().x;
					jointVal["localAnchorB"]["y"] = specJoint->GetLocalAnchorB().y;
					
					jointVal["maxLength"] = specJoint->GetMaxLength();
				}
				break;
			case e_motorJoint:
				{
					b2MotorJoint* specJoint = (b2MotorJoint*)joint;

					jointVal["linearOffset"]["x"] = specJoint->GetLinearOffset().x;
					jointVal["linearOffset"]["y"] = specJoint->GetLinearOffset().y;

					jointVal["angularOffset"] = specJoint->GetAngularOffset();
					
					jointVal["maxForce"] = specJoint->GetMaxForce();
					jointVal["maxTorque"] = specJoint->GetMaxTorque();

					jointVal["correctionFactor"] = specJoint->GetCorrectionFactor();
				}
				break;
			default:
				fprintf(stderr, "serialise: unknown joint type %d\n", joint->GetType());
				break;
			}

			currentJoint++;
		}

		// Save the body that's being followed (if any)
		root["camFollowOn"] = vehicleSim->_cf->getSelected();
		if (vehicleSim->_camFollow != nullptr)
			root["camFollow"] = bodyIds[vehicleSim->_camFollow];

		// Serialise root
		return writer.write(root);
	}

	void Serialiser::deserialiseWorld(VehicleSim* vehicleSim, b2World* world, std::string string)
	{
		std::map<int, b2Body*> bodyMap;

		// Create reader
		Json::Reader reader;

		// Create root
		Json::Value root;

		// Parse file
		bool success = reader.parse(string, root);

		if (!success)
		{
			fprintf(stderr, "Failed to load world\n");
			return;
		}

		// Load drawing settings
		if (root["drawBodies"].asBool())
			vehicleSim->_worldRenderer.AppendFlags(b2Draw::e_shapeBit);
		else
			vehicleSim->_worldRenderer.ClearFlags(b2Draw::e_shapeBit);

		if (root["drawJoints"].asBool())
			vehicleSim->_worldRenderer.AppendFlags(b2Draw::e_jointBit);
		else
			vehicleSim->_worldRenderer.ClearFlags(b2Draw::e_jointBit);

		if (root["drawAABBs"].asBool())
			vehicleSim->_worldRenderer.AppendFlags(b2Draw::e_aabbBit);
		else
			vehicleSim->_worldRenderer.ClearFlags(b2Draw::e_aabbBit);

		// Load gravity
		b2Vec2 gravity;
		gravity.x = root["gravity"]["x"].asFloat();
		gravity.y = root["gravity"]["y"].asFloat();

		vehicleSim->_camera.setPos(glm::vec3(root["camPos"]["x"].asFloat(),
			root["camPos"]["y"].asFloat(), 0));

		vehicleSim->_orthoScale = root["camOrthoScale"].asFloat();

		world->SetGravity(gravity);

		// Load bodies
		Json::Value bodies = root["bodies"];

		const int bodyCount = root["bodyCount"].asInt();
		
		for (int i = 0; i < bodyCount; ++i)
		{
			Json::Value bodyVal = bodies[i];

			// Create body
			b2BodyDef bodyDef;
			bodyDef.type = (b2BodyType)bodyVal["type"].asInt();
			bodyDef.position.x = bodyVal["position"]["x"].asFloat();
			bodyDef.position.y = bodyVal["position"]["y"].asFloat();
			bodyDef.angle = bodyVal["angle"].asFloat();
			bodyDef.linearVelocity.x = bodyVal["linearVelocity"]["x"].asFloat();
			bodyDef.linearVelocity.y = bodyVal["linearVelocity"]["y"].asFloat();
			bodyDef.angularVelocity = bodyVal["angularVelocity"].asFloat();
			bodyDef.linearDamping = bodyVal["linearDamping"].asFloat();
			bodyDef.angularDamping = bodyVal["angularDamping"].asFloat();
			bodyDef.allowSleep = bodyVal["allowSleep"].asBool();
			bodyDef.fixedRotation = bodyVal["fixedRotation"].asBool();
			bodyDef.bullet = bodyVal["bullet"].asBool();
			bodyDef.active = bodyVal["active"].asBool();
			bodyDef.gravityScale = bodyVal["gravityScale"].asFloat();

			b2Body* body = world->CreateBody(&bodyDef);
			bodyMap[i] = body;

			const int fixtureCount = bodyVal["fixtureCount"].asInt();

			for (int j = 0; j < fixtureCount; ++j)
			{
				Json::Value fixtureVal = bodyVal["fixtures"][j];
				Json::Value shapeVal = fixtureVal["shape"];

				// Set up fixture def
				b2FixtureDef fixtureDef;
				fixtureDef.friction = fixtureVal["friction"].asFloat();
				fixtureDef.restitution = fixtureVal["restitution"].asFloat();
				fixtureDef.density = fixtureVal["density"].asFloat();
				fixtureDef.isSensor = fixtureVal["isSensor"].asBool();
				
				fixtureDef.filter.categoryBits = fixtureVal["filter"]["categoryBits"].asInt();
				fixtureDef.filter.maskBits = fixtureVal["filter"]["maskBits"].asInt();
				fixtureDef.filter.groupIndex = fixtureVal["filter"]["groupIndex"].asInt();

				// Set up shape def
				int shapeType = shapeVal["type"].asInt();
				switch (shapeType)
				{
				case b2Shape::e_circle:
					{
						// Create shape
						b2CircleShape shapeDef;
						shapeDef.m_p.Set(shapeVal["position"]["x"].asFloat(),
							shapeVal["position"]["y"].asFloat());
						shapeDef.m_radius = shapeVal["radius"].asFloat();

						// Create fixture and add it to the body
						fixtureDef.shape = &shapeDef;

						body->CreateFixture(&fixtureDef);
					}
					break;
				case b2Shape::e_polygon:
					{
						// Create shape
						b2PolygonShape shapeDef;

						// Get vertex count
						int vertexCount = shapeVal["vertexCount"].asInt();
						
						// Create vertex array
						b2Vec2* vertices = new b2Vec2[vertexCount];

						for (int k = 0; k < vertexCount; ++k)
						{
							vertices[k].x = shapeVal["vertices"][k]["x"].asFloat();
							vertices[k].y = shapeVal["vertices"][k]["y"].asFloat();
						}

						// Set up shape
						shapeDef.Set(vertices, vertexCount);

						// Create fixture and add it to the body
						fixtureDef.shape = &shapeDef;

						body->CreateFixture(&fixtureDef);

						// Delete vertex array
						delete vertices;
					}
					break;
				default:
					fprintf(stderr, "deserialise: unknown shape type %d\n", shapeType);
					break;
				}
			}
		}

		// Load joints
		int jointCount = root["jointCount"].asInt();

		Json::Value joints = root["joints"];

		for (int i = 0; i < jointCount; ++i)
		{
			b2Joint* joint = nullptr;

			// Get current joint
			Json::Value jointVal = joints[i];

			// Get joint type
			b2JointType jointType = (b2JointType)jointVal["jointType"].asInt();

			// Load common values
			b2Body* bodyA = bodyMap[jointVal["bodyA"].asInt()];
			b2Body* bodyB = bodyMap[jointVal["bodyB"].asInt()];

			bool collideConnected = jointVal["collideConnected"].asBool();

			// Load per-type values
			switch (jointType)
			{
			case e_revoluteJoint:
				{
					b2RevoluteJointDef jointDef;

					jointDef.bodyA = bodyA;
					jointDef.bodyB = bodyB;
					jointDef.collideConnected = collideConnected;

					jointDef.localAnchorA =
						b2Vec2(jointVal["localAnchorA"]["x"].asFloat(),
						jointVal["localAnchorA"]["y"].asFloat());

					jointDef.localAnchorB =
						b2Vec2(jointVal["localAnchorB"]["x"].asFloat(),
						jointVal["localAnchorB"]["y"].asFloat());

					jointDef.referenceAngle = jointVal["referenceAngle"].asFloat();

					jointDef.enableLimit = jointVal["enableLimit"].asBool();

					jointDef.upperAngle = jointVal["upperAngle"].asFloat();
					jointDef.lowerAngle = jointVal["lowerAngle"].asFloat();

					jointDef.enableMotor = jointVal["enableMotor"].asBool();
					jointDef.motorSpeed = jointVal["motorSpeed"].asFloat();
					jointDef.maxMotorTorque = jointVal["maxMotorTorque"].asFloat();

					joint = world->CreateJoint(&jointDef);
				}
				break;
			case e_prismaticJoint:
				{
					b2PrismaticJointDef jointDef;

					jointDef.bodyA = bodyA;
					jointDef.bodyB = bodyB;
					jointDef.collideConnected = collideConnected;

					jointDef.localAnchorA =
						b2Vec2(jointVal["localAnchorA"]["x"].asFloat(),
						jointVal["localAnchorA"]["y"].asFloat());

					jointDef.localAnchorB =
						b2Vec2(jointVal["localAnchorB"]["x"].asFloat(),
						jointVal["localAnchorB"]["y"].asFloat());

					jointDef.localAxisA =
						b2Vec2(jointVal["localAxisA"]["x"].asFloat(),
						jointVal["localAxisA"]["y"].asFloat());

					jointDef.referenceAngle = jointVal["referenceAngle"].asFloat();

					jointDef.enableLimit = jointVal["enableLimit"].asBool();

					jointDef.upperTranslation = jointVal["upperTranslation"].asFloat();
					jointDef.lowerTranslation = jointVal["lowerTranslation"].asFloat();

					jointDef.enableMotor = jointVal["enableMotor"].asBool();
					jointDef.motorSpeed = jointVal["motorSpeed"].asFloat();
					jointDef.maxMotorForce = jointVal["maxMotorForce"].asFloat();

					joint = world->CreateJoint(&jointDef);
				}
				break;
			case e_distanceJoint:
				{
					b2DistanceJointDef jointDef;

					jointDef.bodyA = bodyA;
					jointDef.bodyB = bodyB;
					jointDef.collideConnected = collideConnected;

					jointDef.localAnchorA =
						b2Vec2(jointVal["localAnchorA"]["x"].asFloat(),
						jointVal["localAnchorA"]["y"].asFloat());

					jointDef.localAnchorB =
						b2Vec2(jointVal["localAnchorB"]["x"].asFloat(),
						jointVal["localAnchorB"]["y"].asFloat());

					jointDef.length = jointVal["length"].asFloat();

					jointDef.frequencyHz = jointVal["frequency"].asFloat();
					jointDef.dampingRatio = jointVal["dampingRatio"].asFloat();

					joint = world->CreateJoint(&jointDef);
				}
				break;
			case e_pulleyJoint:
				{
					b2PulleyJointDef jointDef;

					jointDef.bodyA = bodyA;
					jointDef.bodyB = bodyB;
					jointDef.collideConnected = collideConnected;

					jointDef.groundAnchorA =
						b2Vec2(jointVal["groundAnchorA"]["x"].asFloat(),
						jointVal["groundAnchorA"]["y"].asFloat());

					jointDef.groundAnchorB =
						b2Vec2(jointVal["groundAnchorB"]["x"].asFloat(),
						jointVal["groundAnchorB"]["y"].asFloat());

					jointDef.localAnchorA =
						b2Vec2(jointVal["localAnchorA"]["x"].asFloat(),
						jointVal["localAnchorA"]["y"].asFloat());

					jointDef.localAnchorB =
						b2Vec2(jointVal["localAnchorB"]["x"].asFloat(),
						jointVal["localAnchorB"]["y"].asFloat());

					jointDef.lengthA = jointVal["lengthA"].asFloat();
					jointDef.lengthB = jointVal["lengthB"].asFloat();

					jointDef.ratio = jointVal["ratio"].asFloat();

					joint = world->CreateJoint(&jointDef);
				}
				break;
			case e_mouseJoint:
				{
					b2MouseJointDef jointDef;

					jointDef.bodyA = bodyA;
					jointDef.bodyB = bodyB;
					jointDef.collideConnected = collideConnected;

					jointDef.target =
						b2Vec2(jointVal["target"]["x"].asFloat(),
						jointVal["target"]["y"].asFloat());

					jointDef.maxForce = jointVal["maxForce"].asFloat();
					
					jointDef.frequencyHz = jointVal["frequency"].asFloat();

					jointDef.dampingRatio = jointVal["dampingRatio"].asFloat();

					joint = world->CreateJoint(&jointDef);
				}
				break;
			case e_wheelJoint:
				{
					b2WheelJointDef jointDef;

					jointDef.bodyA = bodyA;
					jointDef.bodyB = bodyB;
					jointDef.collideConnected = collideConnected;

					jointDef.localAnchorA =
						b2Vec2(jointVal["localAnchorA"]["x"].asFloat(),
						jointVal["localAnchorA"]["y"].asFloat());

					jointDef.localAnchorB =
						b2Vec2(jointVal["localAnchorB"]["x"].asFloat(),
						jointVal["localAnchorB"]["y"].asFloat());

					jointDef.localAxisA =
						b2Vec2(jointVal["localAxisA"]["x"].asFloat(),
						jointVal["localAxisA"]["y"].asFloat());

					jointDef.enableMotor = jointVal["enableMotor"].asBool();
					jointDef.motorSpeed = jointVal["motorSpeed"].asFloat();
					jointDef.maxMotorTorque = jointVal["maxMotorTorque"].asFloat();

					jointDef.frequencyHz = jointVal["frequency"].asFloat();
					jointDef.dampingRatio = jointVal["dampingRatio"].asFloat();

					joint = world->CreateJoint(&jointDef);
				}
				break;
			case e_weldJoint:
				{
					b2WeldJointDef jointDef;

					jointDef.bodyA = bodyA;
					jointDef.bodyB = bodyB;
					jointDef.collideConnected = collideConnected;

					jointDef.localAnchorA =
						b2Vec2(jointVal["localAnchorA"]["x"].asFloat(),
						jointVal["localAnchorA"]["y"].asFloat());

					jointDef.localAnchorB =
						b2Vec2(jointVal["localAnchorB"]["x"].asFloat(),
						jointVal["localAnchorB"]["y"].asFloat());

					jointDef.referenceAngle = jointVal["referenceAngle"].asFloat();

					jointDef.frequencyHz = jointVal["frequency"].asFloat();
					jointDef.dampingRatio = jointVal["dampingRatio"].asFloat();

					joint = world->CreateJoint(&jointDef);
				}
				break;
			case e_frictionJoint:
				{
					b2FrictionJointDef jointDef;

					jointDef.bodyA = bodyA;
					jointDef.bodyB = bodyB;
					jointDef.collideConnected = collideConnected;

					jointDef.localAnchorA =
						b2Vec2(jointVal["localAnchorA"]["x"].asFloat(),
						jointVal["localAnchorA"]["y"].asFloat());

					jointDef.localAnchorB =
						b2Vec2(jointVal["localAnchorB"]["x"].asFloat(),
						jointVal["localAnchorB"]["y"].asFloat());

					jointDef.maxForce = jointVal["maxForce"].asFloat();
					jointDef.maxTorque = jointVal["maxTorque"].asFloat();

					joint = world->CreateJoint(&jointDef);
				}
				break;
			case e_ropeJoint:
				{
					b2RopeJointDef jointDef;

					jointDef.bodyA = bodyA;
					jointDef.bodyB = bodyB;
					jointDef.collideConnected = collideConnected;

					jointDef.localAnchorA =
						b2Vec2(jointVal["localAnchorA"]["x"].asFloat(),
						jointVal["localAnchorA"]["y"].asFloat());

					jointDef.localAnchorB =
						b2Vec2(jointVal["localAnchorB"]["x"].asFloat(),
						jointVal["localAnchorB"]["y"].asFloat());

					jointDef.maxLength = jointVal["maxLength"].asFloat();

					joint = world->CreateJoint(&jointDef);
				}
				break;
			case e_motorJoint:
				{
					b2MotorJointDef jointDef;

					jointDef.bodyA = bodyA;
					jointDef.bodyB = bodyB;
					jointDef.collideConnected = collideConnected;

					jointDef.linearOffset =
						b2Vec2(jointVal["linearOffset"]["x"].asFloat(),
						jointVal["linearOffset"]["y"].asFloat());

					jointDef.angularOffset = jointVal["angularOffset"].asFloat();

					jointDef.maxForce = jointVal["maxForce"].asFloat();
					jointDef.maxTorque = jointVal["maxTorque"].asFloat();

					jointDef.correctionFactor = jointVal["correctionFactor"].asFloat();
					
					joint = world->CreateJoint(&jointDef);
				}
				break;
			default:
				fprintf(stderr, "deserialise: unknown joint type %d\n", jointType);
				break;
			}
			
			if (joint != nullptr)
			{
				if (jointVal["activateKey"].isInt())
					joint->SetUserData(new int(jointVal["activateKey"].asInt()));

				if (jointVal["mo"]["set"].asBool())
				{
					MotorInput mo;
					
					mo.setEnabled(jointVal["mo"]["enabled"].asBool());
					mo.setMaxForce(jointVal["mo"]["maxForce"].asFloat());
					mo.setSpeed(jointVal["mo"]["speed"].asFloat());
					mo.setForwardKey(jointVal["mo"]["forwardButton"].asInt());
					mo.setReverseKey(jointVal["mo"]["reverseButton"].asInt());

					joint->SetUserData(new MotorInput(mo));
				}
			}
		}

		// Load the body that's being followed (if any)
		vehicleSim->_cf->setSelected(root["camFollowOn"].asBool());
		if (root.isMember("camFollow"))
		{
			vehicleSim->_camFollow = bodyMap[root["camFollow"].asInt()];
		}
	}

	void Serialiser::destroyWorld(VehicleSim* vehicleSim, b2World* world)
	{
		vehicleSim->_camFollow = nullptr;

		b2Body* body;
		while ((body = world->GetBodyList()) != 0)
		{
			world->DestroyBody(body);
		}
	}
}
